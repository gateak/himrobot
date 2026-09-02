"""``Himrobot-WBC-G1-v0`` -- the Isaac Lab manager-based RL task for the
whole-body controller (WBC), docs/design/whole_body_v1.md secs 2, 3.

Requires isaaclab (real, on the box, or stubbed -- see
tests/test_wbc_isaac_stub.py, the same stub-gate technique
work/pace/g1_port/stub_import_smoke.py uses for the PACE port). This is the
ONLY file in the package that constructs real isaaclab manager-cfg objects;
every ``mdp/*`` module stays import-safe with plain torch alone (see the
``mdp`` package docstring) and is only *wired in* here via ``func=``/
``class_type=`` references.

Nothing here has ever been run against a real Isaac Lab session -- every
attribute path against isaaclab's own base classes (InteractiveSceneCfg,
ManagerBasedRLEnvCfg, ObservationGroupCfg, ActionTerm, CommandTerm, ...) is
this session's best-effort reading of the v2.1.0 API (the cached reference
files under work/pace/g1_port/reference/ and BeyondMimic's own
config/g1/*.py, per docs/design/whole_body_v1.md's own citations), not a
verified fact. See docs/design/whole_body_v1.md sec 11 and
docs/runbooks/whole_body_box.md for what a real box must check first.
"""
from __future__ import annotations

import torch

from himrobot.sim import scene
from himrobot.wbc.isaac import assets
from himrobot.wbc.isaac.mdp import actions, commands, observations, rewards, terminations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

import gymnasium

TASK_ID = "Himrobot-WBC-G1-v0"

SIM_DT = scene.SIM_DT           # 0.002 s -- the evaluator's own physics timestep
DECIMATION = scene.DECIMATION   # 10 -- 0.002 * 10 = 0.02 s = 50 Hz, the evaluator's control rate
EPISODE_LENGTH_S = terminations.EPISODE_LENGTH_STEPS * SIM_DT * DECIMATION  # 3.0 s
NUM_ENVS = 4096
ENV_SPACING = 2.5

# The three named swing libraries (docs/design/whole_body_v1.md sec 4.1) plus
# the held-out eval-600 bank (sec 4.2, never sampled in training) -- box-side
# default paths; a run overrides via SwingLibraryCommandCfg.library_paths.
DEFAULT_LIBRARY_PATHS = {
    "nominal": "/workspace/motions/wbc/nominal.npz",
    "wide": "/workspace/motions/wbc/wide.npz",
    "slow": "/workspace/motions/wbc/slow.npz",
}


# ---------------------------------------------------------------------------
# Command: the pre-recorded swing library standing in for the hitter.
# ---------------------------------------------------------------------------
class SwingLibraryCommand(CommandTerm):
    """Plays back ``mdp.commands`` swing libraries per env instead of a live
    IK solve (docs/design/whole_body_v1.md sec 4.1, sec 6.1). Exposes named
    tensors (u_now, u_p5, u_p10, u_p15, u_p25, clock, leg_default,
    serve_scale) read directly off the term instance -- observations.py and
    the reward functions below fetch it via
    ``env.command_manager.get_term("swing")`` rather than the single-tensor
    ``.command`` property isaaclab's CommandTerm base normally expects
    (kept only for the base class's own bookkeeping)."""

    cfg: "SwingLibraryCommandCfg"

    def __init__(self, cfg: "SwingLibraryCommandCfg", env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        libraries = {label: commands.SwingLibrary.from_npz(path, device=env.device)
                    for label, path in cfg.library_paths.items()}
        self.bank = commands.LibraryBank.build(libraries)
        n = env.num_envs
        self.row_idx = torch.zeros(n, dtype=torch.int64, device=env.device)
        self.time_idx = torch.zeros(n, dtype=torch.int64, device=env.device)
        self.leg_default = torch.as_tensor(cfg.leg_default, dtype=torch.float32,
                                           device=env.device).unsqueeze(0).expand(n, -1).clone()
        self.iteration = 0  # advanced externally by scripts/train.py's curriculum hook
        self.ring = commands.PostSwingRing(cfg.ring_capacity)

    @property
    def command(self) -> torch.Tensor:
        return self.u_now

    @property
    def u_now(self) -> torch.Tensor:
        return commands.gather_u(self.bank.combined.u, self.row_idx, self.time_idx)

    def preview(self, offset: int) -> torch.Tensor:
        return commands.gather_u(self.bank.combined.u, self.row_idx, self.time_idx + offset)

    @property
    def u_p5(self) -> torch.Tensor:
        return self.preview(5)

    @property
    def u_p10(self) -> torch.Tensor:
        return self.preview(10)

    @property
    def u_p15(self) -> torch.Tensor:
        return self.preview(15)

    @property
    def u_p25(self) -> torch.Tensor:
        return self.preview(25)

    @property
    def serve_scale(self) -> torch.Tensor:
        return self.bank.combined.serve_scale[self.row_idx]

    @property
    def clock(self) -> torch.Tensor:
        contact = self.bank.combined.contact_step[self.row_idx]
        start = self.bank.combined.swing_start_step[self.row_idx]
        return commands.swing_clock(self.time_idx, contact, start, SIM_DT * DECIMATION)

    def _update_command(self) -> None:
        self.time_idx += 1

    def _update_metrics(self) -> None:  # pragma: no cover -- isaaclab logging hook, no metrics defined yet
        pass

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """Assign fresh rows to the given envs per the curriculum in
        docs/design/whole_body_v1.md sec 6.1: library-mix-by-iteration for
        row_start/pre_swing draws, the ring buffer for post_swing_ring, idle
        rows read STAND_POSE (no library row at all -- represented here as
        row 0 of "nominal" held at its own step 0, since the combined tensor
        has no dedicated all-zero row; a real recorder-side idle row is a
        WB1 follow-up)."""
        n = int(env_ids.numel())
        if n == 0:
            return
        sources = commands.sample_reset_sources(n, self.ring.filled)
        mix = commands.library_mix_at(self.iteration)
        labels = list(mix.keys())
        weights = torch.tensor([mix[l] for l in labels], dtype=torch.float32)
        for i, env_id in enumerate(env_ids.tolist()):
            source = sources[i]
            if source == "row_start":
                label = labels[int(torch.multinomial(weights, 1).item())]
                self.row_idx[env_id] = self.bank.sample_rows(label, 1)[0]
                self.time_idx[env_id] = 0
            elif source == "pre_swing":
                label = labels[int(torch.multinomial(weights, 1).item())]
                row = self.bank.sample_rows(label, 1)[0]
                self.row_idx[env_id] = row
                start = int(self.bank.combined.swing_start_step[row].item())
                self.time_idx[env_id] = max(0, start - 5)
            elif source == "post_swing_ring":
                sampled = self.ring.sample(1)
                if sampled:
                    self.row_idx[env_id] = sampled[0]["row_idx"]
                    self.time_idx[env_id] = sampled[0]["time_idx"]
                else:  # ring not yet filled -- see sample_reset_sources' own min_fill guard
                    self.row_idx[env_id] = self.bank.sample_rows(labels[0], 1)[0]
                    self.time_idx[env_id] = 0
            else:  # idle
                self.row_idx[env_id] = 0
                self.time_idx[env_id] = int(self.bank.combined.n_steps - 1)  # the row's own post-swing hold


@configclass
class SwingLibraryCommandCfg(CommandTermCfg):
    class_type: type = SwingLibraryCommand
    library_paths: dict = dict(DEFAULT_LIBRARY_PATHS)
    leg_default: tuple = actions.STAND_LEG_DEFAULT
    ring_capacity: int = commands.RING_BUFFER_SIZE
    resampling_time_range: tuple = (1.0e9, 1.0e9)  # rows are resampled on episode reset, never on a timer


# ---------------------------------------------------------------------------
# Action: the 15-dim WBC action -> 29-dim joint target.
# ---------------------------------------------------------------------------
class WBCJointAction(ActionTerm):
    """Wraps ``mdp.actions.joint_target_wbc``. See the module docstring's
    caveat -- unverified against a real Isaac Lab session."""

    cfg: "WBCJointActionCfg"

    def __init__(self, cfg: "WBCJointActionCfg", env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._asset = env.scene[cfg.asset_name]
        self._raw_actions = torch.zeros(env.num_envs, actions.ACTION_DIM, device=env.device)
        self._processed_actions = torch.zeros(env.num_envs, scene.NUM_JOINTS, device=env.device)
        limits = self._asset.data.soft_joint_pos_limits  # (num_envs, 29, 2), isaaclab's own per-joint limits
        self._jnt_low = limits[0, :, 0].clone()
        self._jnt_high = limits[0, :, 1].clone()
        self._leg_default = torch.as_tensor(cfg.leg_default, dtype=torch.float32, device=env.device)

    @property
    def action_dim(self) -> int:
        return actions.ACTION_DIM

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, action: torch.Tensor) -> None:
        self._raw_actions[:] = action
        swing = self._env.command_manager.get_term("swing")
        self._processed_actions[:] = actions.joint_target_wbc(
            self._raw_actions, swing.u_now, self._leg_default, self._jnt_low, self._jnt_high,
        )

    def apply_action(self) -> None:
        self._asset.set_joint_position_target(self._processed_actions)


@configclass
class WBCJointActionCfg(ActionTermCfg):
    class_type: type = WBCJointAction
    asset_name: str = "robot"
    leg_default: tuple = actions.STAND_LEG_DEFAULT


# ---------------------------------------------------------------------------
# Scene: ground plane + the G1_WBC_CFG robot, contact sensor on every body.
# ---------------------------------------------------------------------------
@configclass
class WBCSceneCfg(InteractiveSceneCfg):
    terrain: TerrainImporterCfg = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane",
                                                     collision_group=-1)
    robot: ArticulationCfg = assets.build_g1_wbc_cfg().replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True,
    )
    sky_light: sim_utils.DomeLightCfg = sim_utils.DomeLightCfg(intensity=750.0)


# ---------------------------------------------------------------------------
# Observations.
# ---------------------------------------------------------------------------
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        actor = ObsTerm(func=observations.actor_obs_func)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        actor = ObsTerm(func=observations.actor_obs_func)
        extra = ObsTerm(func=observations.critic_extra_func)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class ActionsCfg:
    wbc_action: WBCJointActionCfg = WBCJointActionCfg(asset_name="robot", leg_default=actions.STAND_LEG_DEFAULT)


@configclass
class CommandsCfg:
    swing: SwingLibraryCommandCfg = SwingLibraryCommandCfg(library_paths=dict(DEFAULT_LIBRARY_PATHS))


# ---------------------------------------------------------------------------
# Rewards -- one RewTerm per rewards.REWARD_TERMS row, docs/design/
# whole_body_v1.md sec 3.3. Each _*_reward function is the isaaclab-facing
# adapter; the actual math is rewards.py's pure kernel.
# ---------------------------------------------------------------------------
def _swing(env):  # noqa: ANN001
    return env.command_manager.get_term("swing")


def _station_reward(env, pre_strike: bool = True):  # noqa: ANN001
    robot = env.scene["robot"]
    kernel = rewards.station_kernel(robot.data.root_pos_w[:, :2], scene.BASE_NOMINAL_POS[:2])
    gate = _swing(env).clock[:, 0] if pre_strike else (1.0 - _swing(env).clock[:, 0])
    return kernel * gate


def _box_excursion_reward(env):  # noqa: ANN001
    robot = env.scene["robot"]
    return rewards.box_excursion_sq(robot.data.root_pos_w[:, :2], scene.BASE_NOMINAL_POS[:2])


def _height_reward(env):  # noqa: ANN001
    return rewards.height_kernel(env.scene["robot"].data.root_pos_w[:, 2])


def _upright_reward(env):  # noqa: ANN001
    gravity_b = observations.body_frame_gravity(env.scene["robot"].data.root_quat_w)
    tilt = torch.arccos(torch.clamp(-gravity_b[:, 2], -1.0, 1.0))
    return rewards.upright_kernel(tilt)


def _heading_reward(env):  # noqa: ANN001
    heading = observations.heading_cos_sin(env.scene["robot"].data.root_quat_w)
    target = torch.zeros_like(heading)
    target[:, 0] = 1.0  # facing +x, scene.py's own "identity orientation faces the table" convention
    return rewards.heading_kernel(rewards.heading_error_rad(heading, target))


def _base_lin_vel_reward(env):  # noqa: ANN001
    return rewards.base_lin_vel_kernel(env.scene["robot"].data.root_lin_vel_w)


def _base_ang_vel_reward(env):  # noqa: ANN001
    return rewards.base_ang_vel_kernel(env.scene["robot"].data.root_ang_vel_b)


def _feet_contact_reward(env):  # noqa: ANN001
    forces = env.scene["contact_forces"].data.net_forces_w_history[:, 0, :, :].norm(dim=-1)
    left, right = forces[:, 0], forces[:, 1]  # ordering per _FEET's body_names regex match order
    return rewards.feet_contact_bonus(left, right)


def _foot_slip_reward(env):  # noqa: ANN001
    robot = env.scene["robot"]
    foot_ids, _ = robot.find_bodies(".*_ankle_roll_link")
    foot_vel_xy = robot.data.body_lin_vel_w[:, foot_ids, :2]
    forces = env.scene["contact_forces"].data.net_forces_w_history[:, 0, :, :].norm(dim=-1)
    in_contact = forces[:, :2] > rewards.FEET_CONTACT_FORCE_N
    return rewards.foot_slip_penalty(foot_vel_xy, in_contact)


def _mocap_leg_prior_reward(env):  # noqa: ANN001
    # Placeholder phase-0 prior until assets/wbc/mocap_leg_prior.npz (WB5)
    # lands: the mean pose (all-zero delta) is a neutral, always-satisfiable
    # target, so this term contributes 1.0 * weight until the real prior is wired in.
    robot = env.scene["robot"]
    q_legs = robot.data.joint_pos[:, :10]  # legs minus the two ankle-roll joints
    return rewards.mocap_leg_prior_kernel(q_legs, torch.zeros_like(q_legs))


def _action_rate_reward(env):  # noqa: ANN001
    return rewards.action_rate_penalty(env.action_manager.action, env.action_manager.prev_action)


def _leg_joint_vel_reward(env):  # noqa: ANN001
    return rewards.joint_vel_penalty(env.scene["robot"].data.joint_vel[:, :12])


def _leg_torque_reward(env):  # noqa: ANN001
    return rewards.torque_penalty(env.scene["robot"].data.applied_torque[:, :12])


def _joint_limit_reward(env):  # noqa: ANN001
    robot = env.scene["robot"]
    limits = robot.data.soft_joint_pos_limits[0]
    return rewards.joint_limit_violation(robot.data.joint_pos, limits[:, 0], limits[:, 1])


def _waist_residual_reward(env):  # noqa: ANN001
    a = env.action_manager.action
    return rewards.waist_residual_penalty(a[:, actions.N_LEG:actions.N_LEG + actions.N_WAIST])


def _undesired_contact_reward(env):  # noqa: ANN001
    forces = env.scene["contact_forces"].data.net_forces_w_history[:, 0, 2:, :].norm(dim=-1)
    return rewards.undesired_contact_penalty(forces)


@configclass
class RewardsCfg:
    station = RewTerm(func=_station_reward, weight=rewards.REWARD_WEIGHTS["station"], params={"pre_strike": True})
    station_post_swing = RewTerm(func=_station_reward, weight=rewards.REWARD_WEIGHTS["station_post_swing"],
                                 params={"pre_strike": False})
    box_excursion = RewTerm(func=_box_excursion_reward, weight=rewards.REWARD_WEIGHTS["box_excursion"])
    height = RewTerm(func=_height_reward, weight=rewards.REWARD_WEIGHTS["height"])
    upright = RewTerm(func=_upright_reward, weight=rewards.REWARD_WEIGHTS["upright"])
    heading = RewTerm(func=_heading_reward, weight=rewards.REWARD_WEIGHTS["heading"])
    base_lin_vel = RewTerm(func=_base_lin_vel_reward, weight=rewards.REWARD_WEIGHTS["base_lin_vel"])
    base_ang_vel = RewTerm(func=_base_ang_vel_reward, weight=rewards.REWARD_WEIGHTS["base_ang_vel"])
    feet_contact = RewTerm(func=_feet_contact_reward, weight=rewards.REWARD_WEIGHTS["feet_contact"])
    foot_slip = RewTerm(func=_foot_slip_reward, weight=rewards.REWARD_WEIGHTS["foot_slip"])
    mocap_leg_prior = RewTerm(func=_mocap_leg_prior_reward, weight=rewards.REWARD_WEIGHTS["mocap_leg_prior"])
    action_rate = RewTerm(func=_action_rate_reward, weight=rewards.REWARD_WEIGHTS["action_rate"])
    leg_joint_vel = RewTerm(func=_leg_joint_vel_reward, weight=rewards.REWARD_WEIGHTS["leg_joint_vel"])
    leg_torque = RewTerm(func=_leg_torque_reward, weight=rewards.REWARD_WEIGHTS["leg_torque"])
    joint_limit = RewTerm(func=_joint_limit_reward, weight=rewards.REWARD_WEIGHTS["joint_limit"])
    waist_residual = RewTerm(func=_waist_residual_reward, weight=rewards.REWARD_WEIGHTS["waist_residual"])
    undesired_contact = RewTerm(func=_undesired_contact_reward, weight=rewards.REWARD_WEIGHTS["undesired_contact"])


# ---------------------------------------------------------------------------
# Terminations.
# ---------------------------------------------------------------------------
def _fell(env):  # noqa: ANN001
    return terminations.fell(env.scene["robot"].data.root_pos_w[:, 2])


def _tilted(env):  # noqa: ANN001
    gravity_b = observations.body_frame_gravity(env.scene["robot"].data.root_quat_w)
    tilt = torch.arccos(torch.clamp(-gravity_b[:, 2], -1.0, 1.0))
    return terminations.tilted(tilt)


def _time_out(env):  # noqa: ANN001
    return terminations.time_out(env.episode_length_buf)


@configclass
class TerminationsCfg:
    fell = DoneTerm(func=_fell)
    tilted = DoneTerm(func=_tilted)
    time_out = DoneTerm(func=_time_out, time_out=True)


# ---------------------------------------------------------------------------
# Env cfg + gym registration.
# ---------------------------------------------------------------------------
@configclass
class HimrobotWBCG1EnvCfg(ManagerBasedRLEnvCfg):
    scene: WBCSceneCfg = WBCSceneCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    episode_length_s: float = EPISODE_LENGTH_S
    # Re-declared here (isaaclab's real ManagerBasedRLEnvCfg carries this field
    # too) so __post_init__ below has something to assign sim.dt/render_interval
    # onto without depending on the parent class actually providing it.
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg()

    def __post_init__(self) -> None:
        # Plain attribute assignment (isaaclab's own idiom, e.g. reference/
        # isaaclab_v2.1.0_g1_rough_env_cfg.py's G1RoughEnvCfg_PLAY.__post_init__:
        # "self.scene.num_envs = 50") rather than constructor kwargs -- these two
        # fields live on isaaclab's own InteractiveSceneCfg base, not on
        # WBCSceneCfg's own declared fields.
        self.scene.num_envs = NUM_ENVS
        self.scene.env_spacing = ENV_SPACING
        self.sim.dt = SIM_DT
        self.decimation = DECIMATION
        self.sim.render_interval = self.decimation


def apply_sim_dt_override(cfg: HimrobotWBCG1EnvCfg, dt: float, decimation: int) -> None:
    """``--sim-dt`` override (docs/design/whole_body_v1.md sec 3.2: the
    fallback if 0.002x10 is too slow on the box -- e.g. 0.004x5, same 50 Hz
    control rate, half the physics substeps)."""
    cfg.sim.dt = dt
    cfg.decimation = decimation
    cfg.sim.render_interval = decimation
    assert abs(dt * decimation - SIM_DT * DECIMATION) < 1e-9, "must keep the 50 Hz control rate"


gymnasium.register(
    id=TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": HimrobotWBCG1EnvCfg},
)
