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
# a freshly-derived actuator table (real MuJoCo introspection of the compiled
# scored scene -- himrobot/wbc/mujoco_mirror.py's own method, WB4, not yet
# landed) fed to assets.py via its env-var override, so this test needs no
# checked-in assets/wbc/actuators_mujoco.json.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def actuator_table(tmp_path_factory):
    import json

    import mujoco
    import numpy as np

    model = scene.build_model(fixed_base=False, paddle=True, robot_table_collision=False)
    table = {f: {} for f in assets.ACTUATOR_FIELDS}
    for name in scene.JOINT_ORDER:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        dof = model.jnt_dofadr[jid]
        act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        assert act_id >= 0
        table["stiffness"][name] = float(model.actuator_gainprm[act_id, 0])
        table["damping"][name] = float(-model.actuator_biasprm[act_id, 2])
        upper = model.jnt_actfrcrange[jid, 1] if model.jnt_actfrclimited[jid] else 1e9
        table["effort_limit"][name] = float(upper)
        table["armature"][name] = float(model.dof_armature[dof])
        table["friction"][name] = float(model.dof_frictionloss[dof])

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
    for name in scene.JOINT_ORDER:
        act = cfg.actuators[name]
        assert act.stiffness == pytest.approx(table["stiffness"][name])
        assert act.damping == pytest.approx(table["damping"][name])
        assert act.effort_limit == pytest.approx(table["effort_limit"][name])
        assert act.armature == pytest.approx(table["armature"][name])
        assert act.friction == pytest.approx(table["friction"][name])
    # 88/139/50/25/5 N.m pattern (docs/design/whole_body_v1.md sec 3.2)
    assert cfg.actuators["left_hip_pitch_joint"].effort_limit == pytest.approx(88.0)
    assert cfg.actuators["left_hip_roll_joint"].effort_limit == pytest.approx(139.0)
    assert cfg.actuators["left_ankle_roll_joint"].effort_limit == pytest.approx(50.0)
    assert cfg.actuators["right_shoulder_pitch_joint"].effort_limit == pytest.approx(25.0)
    assert cfg.actuators["right_wrist_pitch_joint"].effort_limit == pytest.approx(5.0)


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
