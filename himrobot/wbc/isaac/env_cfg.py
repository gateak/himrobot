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

Known gaps against the design (open items, not silently assumed done):
  - sec 6.1's 25 % "policy's own post-swing states" resets: the ring buffer
    exists (mdp.commands.PostSwingRing) but nothing captures or restores
    physical state into it, so that share folds into pre_swing resets.
  - sec 6.2: torso CoM offset, paddle-mass, armature-scale and actuator-
    delay randomisation are not implemented (EventsCfg has friction, link
    mass, PD-gain scale, pushes, reset pose/joint noise).
  - sec 3.1 base linear velocity uses root_lin_vel_w (isaaclab's root COM
    velocity); MuJoCo's qvel[0:3] is the pelvis body origin's -- they differ
    by omega x r_com, well inside the +-0.5 m/s training noise.
"""
from __future__ import annotations

import copy
import os

import numpy as np
import torch

from himrobot.paths import ASSETS_DIR
from himrobot.sim import scene
from himrobot.wbc.isaac import assets
from himrobot.wbc.isaac.mdp import actions, commands, observations, rewards, terminations

import isaaclab.envs.mdp as isaac_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
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
STEP_DT = SIM_DT * DECIMATION
EPISODE_LENGTH_S = terminations.EPISODE_LENGTH_STEPS * STEP_DT  # 3.0 s
NUM_ENVS = 4096
ENV_SPACING = 2.5

# The three named swing libraries (docs/design/whole_body_v1.md sec 4.1) plus
# the held-out eval-600 bank (sec 4.2, never sampled in training) -- box-side
# default paths; a run overrides via SwingLibraryCommandCfg.library_paths.
# serve_scale per label matches sec 4.1: wide is recorded with serve_scale
# 1.3, nominal/slow at the unwidened 1.0 (slow instead scales TT_SWING_SPEED,
# which the library schema does not carry -- himrobot.wbc.replay_library's
# LIBRARY_KEYS docstring).
DEFAULT_LIBRARY_PATHS = {
    "nominal": "/workspace/motions/wbc/nominal.npz",
    "wide": "/workspace/motions/wbc/wide.npz",
    "slow": "/workspace/motions/wbc/slow.npz",
}
DEFAULT_LIBRARY_SERVE_SCALE = {"nominal": 1.0, "wide": 1.3, "slow": 1.0}

# Mocap lower-body prior (WB5, tools/mocap/mocap_leg_prior.py -> assets/wbc/
# mocap_leg_prior.npz, landed): bins_mean (10, 12), phase-binned per-leg-joint
# mean over the 7 play clips. Leg joints EXCLUDING the two ankle joints (least
# trustworthy -- foot penetration up to -0.08 m, docs/design/whole_body_v1.md
# sec 4.4): JOINT_ORDER[0:12] is [L hip_pitch/roll/yaw, L knee, L ankle_pitch/
# roll, R hip_pitch/roll/yaw, R knee, R ankle_pitch/roll], so indices 4, 5,
# 10, 11 are the ankles and are excluded here.
MOCAP_LEG_PRIOR_NPZ = ASSETS_DIR / "wbc" / "mocap_leg_prior.npz"
LEG_NO_ANKLE_IDX = [0, 1, 2, 3, 6, 7, 8, 9]
N_PHASE_BINS = 10


def _load_mocap_leg_prior_bins() -> torch.Tensor | None:
    if not os.path.isfile(MOCAP_LEG_PRIOR_NPZ):
        return None
    z = np.load(str(MOCAP_LEG_PRIOR_NPZ), allow_pickle=False)
    return torch.as_tensor(np.asarray(z["bins_mean"], dtype=np.float32))  # (10, 12)


MOCAP_LEG_PRIOR_BINS = _load_mocap_leg_prior_bins()

FEET_BODY_REGEX = ".*_ankle_roll_link"  # the two feet, same names as the MJCF (mujoco_mirror.fk_truth)


def _resolve_scene_ids(env: ManagerBasedRLEnv) -> dict:
    """Name-resolved index maps the whole task shares (built once, on the
    command term, at manager construction -- the scene is initialised by
    then, isaaclab's own JointAction/SceneEntityCfg resolve the same way):

      joint_ids         articulation joint index of scene.JOINT_ORDER[i]
      leg_joint_ids     joint_ids[:12]
      foot_body_ids     articulation body ids of the two feet (L, R)
      foot_sensor_ids   the same two feet in the contact sensor's own body
                        order (a sensor over "Robot/.*" has its own order)
      nonfoot_sensor_ids  every other sensor body -- "undesired contact"

    A converted USD's joint/body order is PhysX's tree traversal, not the
    SDK order, so nothing in this package may index a 29-vector or a body
    tensor positionally without going through these."""
    robot = env.scene["robot"]
    sensor = env.scene["contact_forces"]
    joint_ids, joint_names = robot.find_joints(list(scene.JOINT_ORDER), preserve_order=True)
    if list(joint_names) != list(scene.JOINT_ORDER) or len(joint_ids) != scene.NUM_JOINTS:
        raise RuntimeError(f"USD joints {joint_names} do not cover scene.JOINT_ORDER one-for-one")
    foot_body_ids, foot_names = robot.find_bodies(FEET_BODY_REGEX)
    if len(foot_body_ids) != 2:
        raise RuntimeError(f"expected two {FEET_BODY_REGEX} bodies, found {foot_names}")
    foot_sensor_ids, _ = sensor.find_bodies(FEET_BODY_REGEX)
    n_sensor_bodies = len(sensor.body_names)
    nonfoot_sensor_ids = [i for i in range(n_sensor_bodies) if i not in foot_sensor_ids]
    return {
        "joint_ids": list(joint_ids),
        "leg_joint_ids": list(joint_ids[:actions.N_LEG]),
        "foot_body_ids": list(foot_body_ids),
        "foot_sensor_ids": list(foot_sensor_ids),
        "nonfoot_sensor_ids": nonfoot_sensor_ids,
    }


# ---------------------------------------------------------------------------
# Command: the pre-recorded swing library standing in for the hitter.
# ---------------------------------------------------------------------------
class SwingLibraryCommand(CommandTerm):
    """Plays back ``mdp.commands`` swing libraries per env instead of a live
    IK solve (docs/design/whole_body_v1.md sec 4.1, sec 6.1). Exposes named
    tensors (u_now, u_p5, u_p10, u_p15, u_p25, t_now/t_contact/speed_buffer/
    t_pre/t_follow, leg_default, serve_scale) read directly off the term
    instance -- observations.py and the reward functions below fetch it via
    ``env.command_manager.get_term("swing")`` rather than the single-tensor
    ``.command`` property isaaclab's CommandTerm base normally expects
    (kept only for the base class's own bookkeeping).

    Every ``u_*`` tensor is the FULL 29-dim target (legs zero-padded,
    mdp.commands.pad29) that mdp.observations / mdp.actions read [12:29]
    of, exactly as himrobot.wbc.replay_library._pad29 feeds the numpy twins.
    Envs flagged ``idle`` read STAND_POSE[12:] on every step (sec 6.1's
    5 % idle resets)."""

    cfg: "SwingLibraryCommandCfg"

    def __init__(self, cfg: "SwingLibraryCommandCfg", env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._env = env  # what isaaclab's CommandTerm base sets too; explicit so reset() never depends on it
        libraries = {
            label: commands.SwingLibrary.from_npz(
                path, label=label, serve_scale=cfg.library_serve_scale.get(label, 1.0), device=env.device,
            )
            for label, path in cfg.library_paths.items()
        }
        self.bank = commands.LibraryBank.build(libraries)
        n = env.num_envs
        self.row_idx = torch.zeros(n, dtype=torch.int64, device=env.device)
        self.time_idx = torch.zeros(n, dtype=torch.int64, device=env.device)
        self.idle = torch.zeros(n, dtype=torch.bool, device=env.device)
        self.leg_default = torch.as_tensor(cfg.leg_default, dtype=torch.float32,
                                           device=env.device).unsqueeze(0).expand(n, -1).clone()
        self.iteration = 0  # advanced by scripts/train.py's curriculum hook (once per PPO update)
        self.ring = commands.PostSwingRing(cfg.ring_capacity)
        ids = _resolve_scene_ids(env)
        self.joint_ids: list[int] = ids["joint_ids"]
        self.leg_joint_ids: list[int] = ids["leg_joint_ids"]
        self.foot_body_ids: list[int] = ids["foot_body_ids"]
        self.foot_sensor_ids: list[int] = ids["foot_sensor_ids"]
        self.nonfoot_sensor_ids: list[int] = ids["nonfoot_sensor_ids"]

    @property
    def command(self) -> torch.Tensor:
        return self.u_now

    def _u17(self, offset: int) -> torch.Tensor:
        u = commands.gather_u(self.bank.u, self.row_idx, self.time_idx + offset)
        return commands.apply_idle(u, self.idle)

    @property
    def u_now(self) -> torch.Tensor:
        return commands.pad29(self._u17(0))

    def preview(self, offset: int) -> torch.Tensor:
        return commands.pad29(self._u17(offset))

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
        return commands.gather_scalar(self.bank.serve_scale, self.row_idx)

    @property
    def t_now(self) -> torch.Tensor:
        """Seconds since the row's own episode clock started (the SAME
        clock t_contact/t_pre/t_follow are recorded against, see
        mdp.commands' module docstring) -- time_idx tracks steps within the
        row's own u[] array from wherever it was reset, so this is valid
        regardless of which step a reset resumed at."""
        return self.time_idx.to(torch.float32) * STEP_DT

    @property
    def t_contact(self) -> torch.Tensor:
        return commands.gather_scalar(self.bank.t_contact, self.row_idx)

    @property
    def t_pre(self) -> torch.Tensor:
        return commands.gather_scalar(self.bank.t_pre, self.row_idx)

    @property
    def t_follow(self) -> torch.Tensor:
        return commands.gather_scalar(self.bank.t_follow, self.row_idx)

    @property
    def speed_buffer(self) -> torch.Tensor:
        return torch.full_like(self.t_contact, commands.SPEED_BUFFER)

    @property
    def in_window(self) -> torch.Tensor:
        """1.0 while the swing plan's window is active -- observations.
        swing_clock's own in-window flag (slot 0), exposed here so reward
        terms (station/station_post_swing) don't recompute the window."""
        return observations.swing_clock(self.t_now, self.t_contact, self.speed_buffer,
                                        self.t_pre, self.t_follow)[:, 0]

    def _update_command(self) -> None:
        self.time_idx += 1

    def _update_metrics(self) -> None:  # pragma: no cover -- isaaclab logging hook, no metrics defined yet
        pass

    def reset(self, env_ids=None):  # noqa: ANN001 -- isaaclab CommandTerm.reset signature
        """Adaptive row sampling (sec 6.1): before the base class resamples
        the rows of the envs being reset, fold whether each one ended in a
        real termination (fell/tilted, not a time-out) into its row's
        failure EMA. The termination manager still holds this step's flags
        at this point (ManagerBasedRLEnv._reset_idx resets it after the
        command manager)."""
        tm = getattr(self._env, "termination_manager", None)
        if env_ids is not None and tm is not None:
            ids = torch.as_tensor(env_ids, dtype=torch.int64, device=self.row_idx.device)
            if int(ids.numel()) > 0:
                failed = tm.terminated[ids] & ~self.idle[ids]
                self.bank.record_outcomes(self.row_idx[ids], failed)
        return super().reset(env_ids)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """Assign fresh rows to the given envs per the curriculum in
        docs/design/whole_body_v1.md sec 6.1: library-mix-by-iteration for
        row_start/pre_swing draws (restricted to the labels this run
        loaded), idle rows flagged so they read STAND_POSE throughout.
        "post_swing_ring" draws only ever reach the ring branch once the
        ring holds RING_MIN_FILL states; nothing pushes into the ring yet
        (capturing and restoring the policy's own post-swing physical state
        is not implemented -- see the module docstring's gaps list), so
        today that share folds into pre_swing via sample_reset_sources."""
        n = int(env_ids.numel())
        if n == 0:
            return
        sources = commands.sample_reset_sources(n, self.ring.filled)
        mix = commands.restrict_mix(commands.library_mix_at(self.iteration), self.bank.labels)
        labels = list(mix.keys())
        weights = torch.tensor([mix[label] for label in labels], dtype=torch.float32)
        jitters = torch.randint(-commands.TIME_JITTER_STEPS, commands.TIME_JITTER_STEPS + 1, (n,)).tolist()
        for i, env_id in enumerate(env_ids.tolist()):
            source = sources[i]
            self.idle[env_id] = False
            if source == "row_start":
                label = labels[int(torch.multinomial(weights, 1).item())]
                self.row_idx[env_id] = self.bank.sample_rows(label, 1)[0]
                self.time_idx[env_id] = max(0, jitters[i])  # sec 6.1 time-jitter, +-3 steps
            elif source == "pre_swing":
                label = labels[int(torch.multinomial(weights, 1).item())]
                row = self.bank.sample_rows(label, 1)[0]
                self.row_idx[env_id] = row
                start = int(self.bank.swing_start_step[row].item())
                latest = max(0, start - 5)  # a random step in [0, swing_start - 5] (sec 6.1)
                self.time_idx[env_id] = max(0, int(torch.randint(0, latest + 1, (1,)).item()) + jitters[i])
            elif source == "post_swing_ring":
                sampled = self.ring.sample(1)
                if sampled:
                    self.row_idx[env_id] = sampled[0]["row_idx"]
                    self.time_idx[env_id] = sampled[0]["time_idx"]
                else:  # ring not yet filled -- see sample_reset_sources' own min_fill guard
                    self.row_idx[env_id] = self.bank.sample_rows(labels[0], 1)[0]
                    self.time_idx[env_id] = 0
            else:  # idle: STAND_POSE on every step, row/time irrelevant
                self.idle[env_id] = True
                self.row_idx[env_id] = 0
                self.time_idx[env_id] = 0


@configclass
class SwingLibraryCommandCfg(CommandTermCfg):
    class_type: type = SwingLibraryCommand
    library_paths: dict = dict(DEFAULT_LIBRARY_PATHS)
    library_serve_scale: dict = dict(DEFAULT_LIBRARY_SERVE_SCALE)
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
        self._env = env  # isaaclab's ActionTerm base sets it too; explicit for the same reason as the command term
        self._asset = env.scene[cfg.asset_name]
        self._raw_actions = torch.zeros(env.num_envs, actions.ACTION_DIM, device=env.device)
        self._processed_actions = torch.zeros(env.num_envs, scene.NUM_JOINTS, device=env.device)
        # scene.JOINT_ORDER -> articulation joint index (see _resolve_scene_ids):
        # the 29-dim target is assembled in SDK order and written back through
        # this map; the limits are gathered into the same order.
        self._joint_ids, names = self._asset.find_joints(list(scene.JOINT_ORDER), preserve_order=True)
        if list(names) != list(scene.JOINT_ORDER):
            raise RuntimeError(f"USD joints {names} do not cover scene.JOINT_ORDER one-for-one")
        # (29, 2) [low, high] in JOINT_ORDER -- mdp.actions.joint_target_wbc's
        # jnt_range shape (himrobot.rl.env._joint_range's convention).
        self._jnt_range = self._asset.data.soft_joint_pos_limits[0][self._joint_ids].clone()
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
            self._raw_actions, swing.u_now, self._leg_default, self._jnt_range,
        )

    def apply_action(self) -> None:
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)


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
    # A scene entity must be an AssetBaseCfg (prim_path + spawn); a bare
    # light spawn cfg is rejected by InteractiveScene as an unknown entity.
    sky_light: AssetBaseCfg = AssetBaseCfg(prim_path="/World/skyLight",
                                           spawn=sim_utils.DomeLightCfg(intensity=750.0))


# ---------------------------------------------------------------------------
# Observations.
# ---------------------------------------------------------------------------
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # sec 3.1 train-time noise is applied inside actor_obs_func (per-slot
        # uniform table, observations.ACTOR_OBS_BLOCKS), not via isaaclab's
        # own noise cfg, so the numpy twin's NOISE table is the one source.
        actor = ObsTerm(func=observations.actor_obs_func, params={"noise": True})

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        actor = ObsTerm(func=observations.actor_obs_func, params={"noise": False})
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
# Events: episode resets and docs/design/whole_body_v1.md sec 6.2's
# randomisation. isaaclab resets an articulation's pose ONLY through reset-
# mode events (InteractiveScene.reset clears buffers, it does not move the
# robot) -- without reset_base/reset_joints a fallen robot would stay on the
# floor for the rest of training. Ranges follow sec 6.1 (pre-swing pose
# noise, applied to every reset here) and sec 6.2; the terms not present --
# torso CoM offset, paddle-mass, armature scale, actuator delay -- are open
# items, listed in the module docstring. All isaacsim-side functions are
# isaaclab.envs.mdp's own (unverified here, same caveat as the rest).
# ---------------------------------------------------------------------------
@configclass
class EventsCfg:
    reset_base = EventTerm(
        func=isaac_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "yaw": (-0.1, 0.1)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    reset_joints = EventTerm(
        func=isaac_mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (-0.05, 0.05), "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot")},
    )
    physics_material = EventTerm(
        func=isaac_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.0),
            "dynamic_friction_range": (0.3, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    link_mass = EventTerm(
        func=isaac_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "mass_distribution_params": (0.9, 1.1), "operation": "scale"},
    )
    actuator_gains = EventTerm(
        func=isaac_mdp.randomize_actuator_gains,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stiffness_distribution_params": (0.9, 1.1),
                "damping_distribution_params": (0.9, 1.1),
                "operation": "scale"},
    )
    push_robot = EventTerm(
        func=isaac_mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(2.0, 4.0),
        params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3),
                                   "roll": (-0.3, 0.3), "pitch": (-0.3, 0.3), "yaw": (-0.3, 0.3)},
                "asset_cfg": SceneEntityCfg("robot")},
    )


def disable_randomization(cfg: "HimrobotWBCG1EnvCfg") -> None:
    """Exact nominal resets, no pushes, no material/mass/gain jitter -- for
    Gate 0 (scripts/replay_zero_action.py) and the USD parity check, which
    must compare against MuJoCo's own exact straight-leg reset. Works on a
    private copy of the events block so no shared default is mutated."""
    events = copy.deepcopy(cfg.events)
    events.reset_base.params = {**events.reset_base.params, "pose_range": {}}
    events.reset_joints.params = {**events.reset_joints.params, "position_range": (0.0, 0.0)}
    events.physics_material = None
    events.link_mass = None
    events.actuator_gains = None
    events.push_robot = None
    cfg.events = events


# ---------------------------------------------------------------------------
# Rewards -- one RewTerm per rewards.REWARD_TERMS row, docs/design/
# whole_body_v1.md sec 3.3. Each _*_reward function is the isaaclab-facing
# adapter; the actual math is rewards.py's pure kernel.
# ---------------------------------------------------------------------------
def _swing(env):  # noqa: ANN001
    return env.command_manager.get_term("swing")


def _nominal_xy_like(pos_xy: torch.Tensor) -> torch.Tensor:
    """scene.BASE_NOMINAL_POS[:2] as a tensor on pos_xy's device/dtype (a
    CUDA tensor minus a numpy array raises; CPU silently promotes)."""
    return torch.as_tensor(scene.BASE_NOMINAL_POS[:2], dtype=pos_xy.dtype, device=pos_xy.device)


def _station_reward(env, pre_strike: bool = True):  # noqa: ANN001
    robot = env.scene["robot"]
    pos_xy = (robot.data.root_pos_w - env.scene.env_origins)[:, :2]  # env-local, see assets.py init_state
    kernel = rewards.station_kernel(pos_xy, _nominal_xy_like(pos_xy))
    in_window = _swing(env).in_window
    gate = in_window if pre_strike else (1.0 - in_window)
    return kernel * gate


def _box_excursion_reward(env):  # noqa: ANN001
    robot = env.scene["robot"]
    pos_xy = (robot.data.root_pos_w - env.scene.env_origins)[:, :2]  # env-local, see assets.py init_state
    return rewards.box_excursion_sq(pos_xy, _nominal_xy_like(pos_xy))


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


def _contact_forces_n(env) -> torch.Tensor:  # noqa: ANN001
    """(N, n_sensor_bodies) latest net contact force magnitude per body, in
    the contact sensor's own body order (index through swing.*_sensor_ids)."""
    return env.scene["contact_forces"].data.net_forces_w_history[:, 0, :, :].norm(dim=-1)


def _feet_contact_reward(env):  # noqa: ANN001
    swing = _swing(env)
    forces = _contact_forces_n(env)[:, swing.foot_sensor_ids]
    return rewards.feet_contact_bonus(forces[:, 0], forces[:, 1])


def _foot_slip_reward(env):  # noqa: ANN001
    swing = _swing(env)
    foot_vel_xy = env.scene["robot"].data.body_lin_vel_w[:, swing.foot_body_ids, :2]
    in_contact = _contact_forces_n(env)[:, swing.foot_sensor_ids] > rewards.FEET_CONTACT_FORCE_N
    return rewards.foot_slip_penalty(foot_vel_xy, in_contact)


def _mocap_leg_prior_reward(env):  # noqa: ANN001
    """phase = fraction of the way through the swing plan's own window
    (docs/design/whole_body_v1.md sec 3.3: "phase from the swing clock"),
    binned into MOCAP_LEG_PRIOR_BINS' 10 phase bins (tools/mocap/
    mocap_leg_prior.py, WB5). Falls back to an all-zero (STAND_POSE) target
    if assets/wbc/mocap_leg_prior.npz is absent."""
    swing = _swing(env)
    q_legs = env.scene["robot"].data.joint_pos[:, swing.joint_ids][:, LEG_NO_ANKLE_IDX]
    if MOCAP_LEG_PRIOR_BINS is None:
        return rewards.mocap_leg_prior_kernel(q_legs, torch.zeros_like(q_legs))
    window_start = swing.t_contact - swing.speed_buffer - swing.t_pre
    window_end = swing.t_contact + swing.t_follow
    span = torch.clamp(window_end - window_start, min=1e-6)
    phase = torch.clamp((swing.t_now - window_start) / span, 0.0, 1.0)
    bin_idx = torch.clamp((phase * N_PHASE_BINS).to(torch.int64), 0, N_PHASE_BINS - 1)
    prior = MOCAP_LEG_PRIOR_BINS.to(q_legs.device)[bin_idx][:, LEG_NO_ANKLE_IDX]
    return rewards.mocap_leg_prior_kernel(q_legs, prior)


def _action_rate_reward(env):  # noqa: ANN001
    return rewards.action_rate_penalty(env.action_manager.action, env.action_manager.prev_action)


def _leg_joint_vel_reward(env):  # noqa: ANN001
    return rewards.joint_vel_penalty(env.scene["robot"].data.joint_vel[:, _swing(env).leg_joint_ids])


def _leg_torque_reward(env):  # noqa: ANN001
    return rewards.torque_penalty(env.scene["robot"].data.applied_torque[:, _swing(env).leg_joint_ids])


def _joint_limit_reward(env):  # noqa: ANN001
    robot = env.scene["robot"]
    limits = robot.data.soft_joint_pos_limits[0]
    return rewards.joint_limit_violation(robot.data.joint_pos, limits[:, 0], limits[:, 1])


def _waist_residual_reward(env):  # noqa: ANN001
    a = env.action_manager.action
    return rewards.waist_residual_penalty(a[:, actions.N_LEG:actions.N_LEG + actions.N_WAIST])


def _undesired_contact_reward(env):  # noqa: ANN001
    forces = _contact_forces_n(env)[:, _swing(env).nonfoot_sensor_ids]
    return rewards.undesired_contact_penalty(forces)


def _termination_penalty(env):  # noqa: ANN001
    """1.0 on the step a non-time-out termination fires (isaaclab's own
    mdp.is_terminated), x the -10 weight = sec 3.3's "a termination costs
    -10 once"."""
    return env.termination_manager.terminated.float()


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
    termination = RewTerm(func=_termination_penalty, weight=rewards.REWARD_WEIGHTS["termination"])


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
    events: EventsCfg = EventsCfg()
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
