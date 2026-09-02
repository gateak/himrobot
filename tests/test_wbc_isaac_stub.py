"""Mac-side stub-import gate for himrobot.wbc.isaac (no isaaclab anywhere on
this machine). Same technique as work/pace/g1_port/stub_import_smoke.py:
install permissive stand-ins for isaaclab/isaacsim/omni/rsl_rl/gymnasium into
sys.modules, then import env_cfg.py and ppo_cfg.py FOR REAL and construct
the env/PPO cfg objects, letting every ``__post_init__`` run.

What this catches (same as the PACE precedent): NameErrors, typos in
attribute paths, a config field that does not exist on the parent class, any
module-level or __post_init__ expression that raises. What it cannot catch:
anything about real Isaac Lab behaviour -- the stub accepts every call (see
docs/design/whole_body_v1.md sec 11, and PORT.md's identical caveat for the
PACE port).

One deliberate difference from the PACE stub: ``_Any.__init__`` also sets
every keyword argument as a real instance attribute (not just into an opaque
``_kwargs`` dict), so a constructed stub object -- e.g. an ``ImplicitActuatorCfg``
or ``RewardTermCfg`` -- can be introspected for the exact values THIS
package's own code passed in (``cfg.stiffness``, ``term.weight``, ``term.func``,
...), which is exactly what this test needs to check. ``.replace(**kwargs)``
is likewise a real method (isaaclab's own configs support it, and
env_cfg.py's WBCSceneCfg.robot uses it) rather than falling through the
generic stub, which would otherwise discard every other field.

Run with: .venv/bin/python -m pytest tests/test_wbc_isaac_stub.py -v
"""
from __future__ import annotations

import copy
import dataclasses
import importlib
import importlib.abc
import importlib.machinery
import inspect
import sys
import types

import pytest

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# the permissive stand-in (see module docstring for the one deliberate
# enhancement over work/pace/g1_port/stub_import_smoke.py's own version).
# ---------------------------------------------------------------------------
class _AnyMeta(type):
    def __getattr__(cls, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _make(name)


class _Any(metaclass=_AnyMeta):
    """Accepts any construction, attribute, call, index or iteration."""

    def __init__(self, *args, **kwargs):
        object.__setattr__(self, "_args", args)
        object.__setattr__(self, "_kwargs", dict(kwargs))
        for k, v in kwargs.items():
            try:
                object.__setattr__(self, k, v)
            except Exception:  # pragma: no cover -- a reserved/dunder kwarg name
                pass

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        value = _Any()
        object.__setattr__(self, name, value)
        return value

    def __call__(self, *args, **kwargs):
        return _Any(*args, **kwargs)

    def __getitem__(self, key):
        return self.__getattr__(f"_item_{key}")

    def __setitem__(self, key, value):
        object.__setattr__(self, f"_item_{key}", value)

    def __iter__(self):
        return iter(())

    def replace(self, **overrides):
        """dataclasses.replace's equivalent for a stub cfg instance -- keeps
        every original kwarg, overlays ``overrides``. Needed so
        WBCSceneCfg.robot = assets.build_g1_wbc_cfg().replace(prim_path=...)
        does not silently discard the actuator table it carries."""
        merged = dict(self._kwargs)
        merged.update(overrides)
        return _Any(**merged)

    def reset(self, env_ids=None):
        """isaaclab's CommandTerm.reset (a real method, so ``super().reset``
        resolves): resample the given envs, return the metrics dict."""
        fn = getattr(type(self), "_resample_command", None)
        if fn is not None and env_ids is not None:
            fn(self, env_ids)
        return {}

    def __repr__(self):
        return f"<stub {type(self).__name__} {self._kwargs!r}>"


def _make(name: str) -> type:
    return _AnyMeta(name, (_Any,), {})


def configclass(cls):
    """Stand-in for isaaclab.utils.configclass: dataclasses.dataclass
    (kw_only=True) close enough for an import smoke, matching
    work/pace/g1_port/stub_import_smoke.py's own configclass -- the one
    addition is rewriting a literal dict/list/set default (dataclasses
    forbids those outright) to a default_factory, which
    SwingLibraryCommandCfg.library_paths needs."""
    annotations = dict(inspect.get_annotations(cls))
    # isaaclab's real configclass promotes UNANNOTATED class attributes to
    # fields too (this repo's own RewTerm/ObsTerm/DoneTerm entries in
    # env_cfg.py, e.g. "station = RewTerm(...)", follow the exact style the
    # cached reference/isaaclab_v2.1.0_g1_rough_env_cfg.py itself uses) --
    # without this, dataclass() ignores them entirely, so they never show up
    # as instance attributes (vars(instance) would stay empty).
    for attr, value in list(vars(cls).items()):
        if attr.startswith("__") or attr in annotations:
            continue
        # _Any instances are callable (RewTerm/ObsTerm/DoneTerm stubs), but
        # they are config VALUES, not methods -- only a real callable (a
        # method defined on this class, e.g. __post_init__) is skipped.
        if not isinstance(value, _Any) and (
            callable(value) or isinstance(value, (classmethod, staticmethod, property))
        ):
            continue
        annotations[attr] = "typing.Any"
    cls.__annotations__ = annotations

    for field_name in annotations:
        if field_name not in cls.__dict__:
            continue
        default = cls.__dict__[field_name]
        if isinstance(default, dataclasses.Field):
            continue
        if isinstance(default, (dict, list, set)):
            setattr(cls, field_name, dataclasses.field(default_factory=lambda d=default: copy.deepcopy(d)))
    return dataclasses.dataclass(kw_only=True, eq=False, repr=False)(cls)


class _StubModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _make(name)


STUB_ROOTS = {"isaaclab", "isaacsim", "omni", "carb", "gymnasium", "rsl_rl", "isaaclab_rl"}
STUB_EXPORTS = {"isaaclab.utils": {"configclass": configclass}}


def _is_stubbed(name: str) -> bool:
    return any(name == r or name.startswith(r + ".") for r in STUB_ROOTS)


class _StubLoader(importlib.abc.Loader):
    def create_module(self, spec):
        mod = _StubModule(spec.name)
        mod.__path__ = []  # a package, so arbitrary submodules resolve
        for k, v in STUB_EXPORTS.get(spec.name, {}).items():
            setattr(mod, k, v)
        return mod

    def exec_module(self, module):
        pass


class _StubFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if not _is_stubbed(fullname):
            return None
        return importlib.machinery.ModuleSpec(fullname, _StubLoader(), is_package=True)


def install_stubs() -> None:
    sys.meta_path.insert(0, _StubFinder())
    for mod_name in list(sys.modules):
        if _is_stubbed(mod_name):
            del sys.modules[mod_name]  # a real partial import from an earlier test must not linger


# ---------------------------------------------------------------------------
# install, then import the real package modules for real.
# ---------------------------------------------------------------------------
install_stubs()

for mod_name in ("himrobot.wbc.isaac.assets", "himrobot.wbc.isaac.env_cfg", "himrobot.wbc.isaac.ppo_cfg"):
    sys.modules.pop(mod_name, None)

env_cfg = importlib.import_module("himrobot.wbc.isaac.env_cfg")
ppo_cfg = importlib.import_module("himrobot.wbc.isaac.ppo_cfg")
assets = importlib.import_module("himrobot.wbc.isaac.assets")

from himrobot.sim import scene  # noqa: E402
from himrobot.wbc.isaac.mdp import actions, observations, rewards, terminations  # noqa: E402


# ---------------------------------------------------------------------------
# the actuator table: assets/wbc/actuators_mujoco.json is a real, committed
# asset (himrobot.wbc.mujoco_mirror, WB4, landed) -- this fixture just
# recomputes it fresh (mujoco_mirror.build_actuators_json(), the exact
# function that wrote the committed file) so the test does not depend on
# that file being up to date with the current assets/unitree_g1/g1.xml.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def actuator_table(tmp_path_factory):
    import json

    from himrobot.wbc import mujoco_mirror

    table = mujoco_mirror.build_actuators_json()
    path = tmp_path_factory.mktemp("wbc") / "actuators_mujoco.json"
    path.write_text(json.dumps(table))
    return str(path), table


# ---------------------------------------------------------------------------
# 1. actuator mirror.
# ---------------------------------------------------------------------------
def test_actuator_table_has_29_entries_matching_the_mirror(actuator_table, monkeypatch):
    path, table = actuator_table
    monkeypatch.setenv(assets.ACTUATORS_JSON_ENV, path)
    loaded = assets.load_actuator_table()
    assert loaded == table

    cfg = assets.build_g1_wbc_cfg(table=table)
    assert len(cfg.actuators) == 29
    assert set(cfg.actuators) == set(scene.JOINT_ORDER)
    joints = table["joints"]
    for name in scene.JOINT_ORDER:
        act = cfg.actuators[name]
        entry = joints[name]
        assert act.stiffness == pytest.approx(entry["kp"])
        assert act.damping == pytest.approx(entry["kv"])
        assert act.effort_limit == pytest.approx(max(abs(v) for v in entry["effort_limit"]))
        assert act.armature == pytest.approx(entry["armature"])
        assert act.friction == pytest.approx(entry["frictionloss"])
    # 88/139/50/25/5 N.m pattern (docs/design/whole_body_v1.md sec 3.2)
    assert cfg.actuators["left_hip_pitch_joint"].effort_limit == pytest.approx(88.0)
    assert cfg.actuators["left_hip_roll_joint"].effort_limit == pytest.approx(139.0)
    assert cfg.actuators["left_ankle_roll_joint"].effort_limit == pytest.approx(50.0)
    assert cfg.actuators["right_shoulder_pitch_joint"].effort_limit == pytest.approx(25.0)
    assert cfg.actuators["right_wrist_pitch_joint"].effort_limit == pytest.approx(5.0)


def test_actuator_table_matches_the_real_committed_asset():
    """assets/wbc/actuators_mujoco.json itself (no fixture override) loads
    and validates -- the file WB4 actually committed, not just a freshly
    regenerated copy."""
    table = assets.load_actuator_table()
    assert set(table["joints"]) == set(scene.JOINT_ORDER)


def test_actuator_table_missing_file_raises_with_a_helpful_message():
    with pytest.raises(FileNotFoundError, match="mujoco_mirror"):
        assets.load_actuator_table("/no/such/file.json")


# ---------------------------------------------------------------------------
# 2. env cfg construction.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def env_cfg_instance(actuator_table):
    path, _ = actuator_table
    import os
    old = os.environ.get(assets.ACTUATORS_JSON_ENV)
    os.environ[assets.ACTUATORS_JSON_ENV] = path
    try:
        # G1_WBC_CFG (assets.py's module-level constant) was already built at
        # import time above, against whatever actuator table existed then --
        # rebuild env_cfg's scene robot field with the fixture's table so
        # this test is independent of import order.
        cfg = env_cfg.HimrobotWBCG1EnvCfg()
        cfg.scene.robot = assets.build_g1_wbc_cfg().replace(prim_path="{ENV_REGEX_NS}/Robot")
        return cfg
    finally:
        if old is None:
            os.environ.pop(assets.ACTUATORS_JSON_ENV, None)
        else:
            os.environ[assets.ACTUATORS_JSON_ENV] = old


def test_env_cfg_constructs(env_cfg_instance):
    assert env_cfg_instance is not None


def test_gym_id_registered():
    assert env_cfg.TASK_ID == "Himrobot-WBC-G1-v0"


def test_sim_dt_and_decimation(env_cfg_instance):
    assert env_cfg_instance.sim.dt == pytest.approx(0.002)
    assert env_cfg_instance.decimation == 10
    assert env_cfg_instance.sim.dt * env_cfg_instance.decimation == pytest.approx(0.02)  # 50 Hz


def test_scene_num_envs_and_spacing(env_cfg_instance):
    assert env_cfg_instance.scene.num_envs == 4096
    assert env_cfg_instance.scene.env_spacing == pytest.approx(2.5)


def test_observation_group_wiring(env_cfg_instance):
    policy = env_cfg_instance.observations.policy
    critic = env_cfg_instance.observations.critic
    assert policy.actor.func is observations.actor_obs_func
    assert critic.actor.func is observations.actor_obs_func
    assert critic.extra.func is observations.critic_extra_func
    assert policy.enable_corruption is True
    assert critic.enable_corruption is False
    # sec 3.1: the actor sees the noised 141, the (asymmetric) critic the clean one
    assert policy.actor.params == {"noise": True}
    assert critic.actor.params == {"noise": False}


def test_scene_entities_are_asset_cfgs_and_events_reset_the_robot(env_cfg_instance):
    # A bare DomeLightCfg is not a scene entity in isaaclab; it must be wrapped.
    assert env_cfg_instance.scene.sky_light.prim_path == "/World/skyLight"
    assert env_cfg_instance.scene.sky_light.spawn.intensity == pytest.approx(750.0)
    # isaaclab moves an articulation on reset ONLY through reset-mode events.
    ev = env_cfg_instance.events
    assert ev.reset_base.mode == "reset" and ev.reset_joints.mode == "reset"
    assert ev.reset_base.params["pose_range"]["yaw"] == (-0.1, 0.1)
    assert ev.reset_joints.params["position_range"] == (-0.05, 0.05)
    assert ev.push_robot.mode == "interval" and ev.push_robot.interval_range_s == (2.0, 4.0)
    assert ev.physics_material.params["static_friction_range"] == (0.4, 1.0)
    # Gate 0 / parity check: exact nominal resets, no randomisation
    cfg = env_cfg.HimrobotWBCG1EnvCfg()
    env_cfg.disable_randomization(cfg)
    assert cfg.events.reset_base.params["pose_range"] == {}
    assert cfg.events.reset_joints.params["position_range"] == (0.0, 0.0)
    assert cfg.events.push_robot is None and cfg.events.physics_material is None
    assert env_cfg_instance.events.push_robot is not None  # the fixture instance is untouched
    # Dims themselves are checked directly against the tier-1 data (real
    # Isaac Lab computes a term's dim by calling it with a live env, which
    # this stub cannot do -- see the module docstring).
    assert observations.ACTOR_OBS_DIM == 141
    assert observations.CRITIC_OBS_DIM == 176


def test_action_term_wiring_and_scales(env_cfg_instance):
    action_cfg = env_cfg_instance.actions.wbc_action
    assert action_cfg.class_type is env_cfg.WBCJointAction
    assert action_cfg.asset_name == "robot"
    assert tuple(action_cfg.leg_default) == actions.STAND_LEG_DEFAULT
    assert actions.ACTION_DIM == 15
    assert actions.LEG_SCALE == 0.5
    assert actions.WAIST_SCALE == 0.2


def test_reward_terms_match_design_table(env_cfg_instance):
    got = {name: term.weight for name, term in vars(env_cfg_instance.rewards).items()
          if not name.startswith("_")}
    assert got == rewards.REWARD_WEIGHTS


def test_termination_terms_present(env_cfg_instance):
    term_names = {name for name in vars(env_cfg_instance.terminations) if not name.startswith("_")}
    assert term_names == {"fell", "tilted", "time_out"}
    assert env_cfg_instance.terminations.fell.func is env_cfg._fell
    assert env_cfg_instance.terminations.tilted.func is env_cfg._tilted
    assert env_cfg_instance.terminations.time_out.func is env_cfg._time_out
    assert env_cfg_instance.terminations.time_out.time_out is True


def test_command_term_wiring(env_cfg_instance):
    swing_cfg = env_cfg_instance.commands.swing
    assert swing_cfg.class_type is env_cfg.SwingLibraryCommand
    assert tuple(swing_cfg.leg_default) == actions.STAND_LEG_DEFAULT
    assert swing_cfg.ring_capacity == 8192


# ---------------------------------------------------------------------------
# 2b. The command / observation / reward wiring driven through a fake env
#     whose articulation lists its joints in a PERMUTED order (what a USD
#     converted from our MJCF does: PhysX orders joints by tree traversal, not
#     the SDK order) and whose contact sensor does not put the feet at 0/1.
# ---------------------------------------------------------------------------
import numpy as np  # noqa: E402

from himrobot.wbc import obs as wbc_obs  # noqa: E402
from himrobot.wbc.isaac.mdp import commands  # noqa: E402


class _Names:
    def __init__(self, names):
        self.names = list(names)

    def find(self, keys, preserve_order=False):
        import re
        keys = [keys] if isinstance(keys, str) else list(keys)
        if preserve_order:
            ids = [self.names.index(k) for k in keys]
        else:
            ids = [i for i, n in enumerate(self.names) if any(re.fullmatch(k, n) for k in keys)]
        return ids, [self.names[i] for i in ids]


class _FakeRobot:
    def __init__(self, joint_names, body_names, n, device="cpu"):
        self._j, self._b = _Names(joint_names), _Names(body_names)
        self.joint_names, self.body_names = self._j.names, self._b.names
        self.data = types.SimpleNamespace()
        rng = np.random.default_rng(0)
        q = rng.normal(size=(n, 4)); q /= np.linalg.norm(q, axis=-1, keepdims=True)
        self.data.root_quat_w = torch.as_tensor(q, dtype=torch.float32)
        self.data.root_ang_vel_b = torch.as_tensor(rng.normal(size=(n, 3)), dtype=torch.float32)
        self.data.root_lin_vel_w = torch.as_tensor(rng.normal(size=(n, 3)), dtype=torch.float32)
        self.data.root_pos_w = torch.as_tensor(scene.BASE_NOMINAL_POS + rng.normal(size=(n, 3)) * 0.05,
                                               dtype=torch.float32)
        self.data.joint_pos = torch.as_tensor(rng.normal(size=(n, 29)) * 0.3, dtype=torch.float32)
        self.data.joint_vel = torch.as_tensor(rng.normal(size=(n, 29)), dtype=torch.float32)
        self.data.applied_torque = torch.as_tensor(rng.normal(size=(n, 29)), dtype=torch.float32)
        self.data.body_lin_vel_w = torch.as_tensor(rng.normal(size=(n, len(body_names), 3)), dtype=torch.float32)
        lim = torch.stack([torch.full((29,), -2.0), torch.full((29,), 2.0)], dim=-1)
        self.data.soft_joint_pos_limits = lim.unsqueeze(0).expand(n, -1, -1).clone()
        self.targets = None

    def find_joints(self, keys, preserve_order=False):
        return self._j.find(keys, preserve_order)

    def find_bodies(self, keys, preserve_order=False):
        return self._b.find(keys, preserve_order)

    def set_joint_position_target(self, target, joint_ids=None):
        self.targets = (target.clone(), list(joint_ids) if joint_ids is not None else None)


class _FakeSensor:
    def __init__(self, body_names, n):
        self._b = _Names(body_names)
        self.body_names = self._b.names
        forces = torch.zeros(n, 3, len(body_names), 3)
        self.data = types.SimpleNamespace(net_forces_w_history=forces)

    def find_bodies(self, keys, preserve_order=False):
        return self._b.find(keys, preserve_order)


def _permuted_joint_names():
    # breadth-first-ish interleave: L/R legs alternate, waist and arms mixed in
    order = scene.JOINT_ORDER
    perm = []
    for i in range(6):
        perm += [order[i], order[6 + i]]
    perm += [order[12]] + [order[15 + i] for i in range(7)] + order[13:15] + [order[22 + i] for i in range(7)]
    assert sorted(perm) == sorted(order) and perm != list(order)
    return perm


def _fake_env(n, lib_path):
    joint_names = _permuted_joint_names()
    body_names = ["pelvis", "torso_link", "left_ankle_roll_link", "paddle", "right_ankle_roll_link", "head"]
    sensor_names = ["torso_link", "pelvis", "right_ankle_roll_link", "head", "left_ankle_roll_link", "paddle"]
    robot = _FakeRobot(joint_names, body_names, n)
    sensor = _FakeSensor(sensor_names, n)
    env = types.SimpleNamespace(num_envs=n, device="cpu")
    env.scene = {"robot": robot, "contact_forces": sensor}
    env.scene_env_origins = None
    class _Scene(dict):
        env_origins = torch.zeros(n, 3)
    env.scene = _Scene(env.scene)
    env.action_manager = types.SimpleNamespace(action=torch.zeros(n, 15), prev_action=torch.zeros(n, 15))
    env.termination_manager = types.SimpleNamespace(terminated=torch.zeros(n, dtype=torch.bool))
    cfg = env_cfg.SwingLibraryCommandCfg(library_paths={"eval": lib_path})
    swing = env_cfg.SwingLibraryCommand(cfg, env)
    env.command_manager = types.SimpleNamespace(get_term=lambda name: swing)
    return env, swing, robot, sensor


@pytest.fixture(scope="module")
def eval_library(tmp_path_factory):
    rng = np.random.default_rng(3)
    r, t = 5, 40
    u = np.tile(scene.STAND_POSE[12:], (r, t, 1)).astype(np.float32)
    u[:, 10:30, :] += rng.normal(size=(r, 20, 17)).astype(np.float32) * 0.3  # swings from step 10
    path = tmp_path_factory.mktemp("lib") / "eval.npz"
    np.savez(path, u=u, swing_start_step=np.full(r, 10, np.int32), contact_step=np.full(r, 20, np.int32),
             t_contact=np.full(r, 0.4, np.float32), t_pre=np.full(r, 0.15, np.float32),
             t_follow=np.full(r, 0.1, np.float32), serve_pos=np.zeros((r, 3), np.float32),
             serve_vel=np.zeros((r, 3), np.float32), welded_legal=np.ones(r, bool), welded_contact=np.ones(r, bool))
    return str(path), u


def test_command_term_resolves_permuted_joints_and_sensor_feet(eval_library):
    path, _ = eval_library
    env, swing, robot, sensor = _fake_env(4, path)
    assert [robot.joint_names[i] for i in swing.joint_ids] == list(scene.JOINT_ORDER)
    assert swing.joint_ids != list(range(29))  # the permutation is real
    assert [robot.body_names[i] for i in swing.foot_body_ids] == ["left_ankle_roll_link", "right_ankle_roll_link"]
    assert [sensor.body_names[i] for i in swing.foot_sensor_ids] == ["right_ankle_roll_link", "left_ankle_roll_link"]
    assert sorted(swing.foot_sensor_ids + swing.nonfoot_sensor_ids) == list(range(6))


def test_command_term_u_is_29_wide_and_a_single_unscheduled_library_resamples(eval_library):
    path, u = eval_library
    env, swing, *_ = _fake_env(4, path)
    swing._resample_command(torch.arange(4))  # "eval" is in no schedule entry -> uniform fallback
    assert swing.u_now.shape == (4, 29) and swing.u_p10.shape == (4, 29)
    np.testing.assert_allclose(swing.u_now[:, :12].numpy(), 0.0)
    for i in range(4):
        if not swing.idle[i]:
            row, t = int(swing.row_idx[i]), min(int(swing.time_idx[i]), 39)
            np.testing.assert_allclose(swing.u_now[i, 12:].numpy(), u[row, t], atol=1e-6)
    swing.idle[:] = True
    np.testing.assert_allclose(swing.u_now[:, 12:].numpy(), np.tile(scene.STAND_POSE[12:], (4, 1)), atol=1e-6)


def test_actor_obs_func_matches_numpy_flatten_through_the_joint_permutation(eval_library):
    path, u = eval_library
    n = 6
    env, swing, robot, _ = _fake_env(n, path)
    swing.row_idx[:] = torch.arange(n) % 5
    swing.time_idx[:] = torch.tensor([0, 5, 12, 20, 33, 39])
    swing.idle[:] = False
    env.action_manager.action = torch.as_tensor(np.random.default_rng(9).uniform(-1, 1, (n, 15)), dtype=torch.float32)

    got = observations.actor_obs_func(env, noise=False).numpy()
    assert got.shape == (n, 141)
    ids = swing.joint_ids
    for i in range(n):
        obs_dict = {
            "base.orientation": robot.data.root_quat_w[i].double().numpy(),
            "base.angular_velocity": robot.data.root_ang_vel_b[i].double().numpy(),
            "base.linear_velocity": robot.data.root_lin_vel_w[i].double().numpy(),
            "base.position": robot.data.root_pos_w[i].double().numpy(),
            "robot.joint_position": robot.data.joint_pos[i].double().numpy()[ids],   # SDK order
            "robot.joint_velocity": robot.data.joint_vel[i].double().numpy()[ids],
            "time_s": float(swing.time_idx[i]) * env_cfg.STEP_DT,
        }
        row, t = int(swing.row_idx[i]), int(swing.time_idx[i])
        u29 = lambda k: np.concatenate([np.zeros(12), u[row, min(t + k, 39)]])  # noqa: E731
        plan = {"t_contact": 0.4, "speed_buffer": 0.0, "t_pre": 0.15, "t_follow": 0.1}
        want = wbc_obs.flatten(obs_dict, env.action_manager.action[i].double().numpy(), u29(0), u29(5), u29(10),
                               plan, scene.STAND_POSE[:12])
        np.testing.assert_allclose(got[i], want, atol=2e-5)
    noised = observations.actor_obs_func(env, noise=True).numpy()
    assert noised.shape == (n, 141) and not np.allclose(noised[:, :3], got[:, :3])
    np.testing.assert_allclose(noised[:, 72:], got[:, 72:])  # prev_action / u / clock slots are never noised


def test_action_term_writes_targets_through_the_joint_map(eval_library):
    path, u = eval_library
    n = 3
    env, swing, robot, _ = _fake_env(n, path)
    swing.row_idx[:] = 0
    swing.time_idx[:] = 15
    swing.idle[:] = False
    term = env_cfg.WBCJointAction(env_cfg.WBCJointActionCfg(asset_name="robot"), env)
    term.process_actions(torch.zeros(n, 15))
    term.apply_action()
    target, joint_ids = robot.targets
    assert joint_ids == swing.joint_ids
    want = np.concatenate([scene.STAND_POSE[:12], u[0, 15]])  # a = 0 -> the hitter's own command, bit for bit
    np.testing.assert_allclose(target.numpy(), np.tile(want, (n, 1)), atol=1e-6)


def test_reward_adapters_use_resolved_ids(eval_library):
    path, _ = eval_library
    n = 2
    env, swing, robot, sensor = _fake_env(n, path)
    # feet in contact in env 0 only (sensor order: feet at 2 and 4), torso touching in env 1
    sensor.data.net_forces_w_history[0, 0, 2, 2] = 50.0
    sensor.data.net_forces_w_history[0, 0, 4, 2] = 50.0
    sensor.data.net_forces_w_history[1, 0, 0, 2] = 50.0
    np.testing.assert_allclose(env_cfg._feet_contact_reward(env).numpy(), [1.0, 0.0])
    np.testing.assert_allclose(env_cfg._undesired_contact_reward(env).numpy(), [0.0, 1.0])
    slip = env_cfg._foot_slip_reward(env).numpy()
    speeds = robot.data.body_lin_vel_w[0, swing.foot_body_ids, :2].norm(dim=-1).sum().item()
    np.testing.assert_allclose(slip, [speeds, 0.0], atol=1e-6)
    leg_ids = swing.leg_joint_ids
    np.testing.assert_allclose(env_cfg._leg_torque_reward(env).numpy(),
                               (robot.data.applied_torque[:, leg_ids] ** 2).sum(-1).numpy(), atol=1e-5)
    env.termination_manager.terminated[1] = True
    np.testing.assert_allclose(env_cfg._termination_penalty(env).numpy(), [0.0, 1.0])
    # reset() folds the termination into the row's failure EMA before resampling
    before = swing.bank.fail_ema.clone()
    swing.reset(torch.tensor([1]))
    assert swing.bank.fail_ema[int(swing.row_idx[1])] >= before[int(swing.row_idx[1])]


# ---------------------------------------------------------------------------
# 3. PPO cfg.
# ---------------------------------------------------------------------------
def test_ppo_cfg_hyperparameters():
    cfg = ppo_cfg.HimrobotWBCPPORunnerCfg()
    assert cfg.num_steps_per_env == 24
    assert cfg.save_interval == 500
    assert cfg.empirical_normalization is True
    assert cfg.zero_init_final_actor_layer is True
    assert list(cfg.policy._kwargs["actor_hidden_dims"]) == [512, 256, 128]
    assert cfg.policy._kwargs["activation"] == "elu"
    assert cfg.algorithm._kwargs["num_learning_epochs"] == 5
    assert cfg.algorithm._kwargs["num_mini_batches"] == 4
    assert cfg.algorithm._kwargs["gamma"] == pytest.approx(0.99)
    assert cfg.algorithm._kwargs["lam"] == pytest.approx(0.95)
    assert cfg.algorithm._kwargs["entropy_coef"] == pytest.approx(0.005)
    assert cfg.algorithm._kwargs["learning_rate"] == pytest.approx(1.0e-3)
    assert cfg.algorithm._kwargs["desired_kl"] == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# 4. PACE / BeyondMimic tasks untouched -- this package adds a new task
#    without disturbing the existing ones.
# ---------------------------------------------------------------------------
def test_pace_task_files_untouched():
    import subprocess

    r = subprocess.run(["git", "status", "--porcelain", "--",
                       "work/pace/g1_port/patch", "docs/TRAINING_PLAN.md"],
                      capture_output=True, text=True, cwd=str(scene.REPO))
    assert r.returncode == 0
    assert r.stdout.strip() == "", f"unexpected changes under PACE-owned paths:\n{r.stdout}"
