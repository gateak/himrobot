"""himrobot: the G1 table-tennis simulation, RL training, evaluation and HIM
serving package.

See himrobot.sim for the model/env definitions, himrobot.policies for the
Policy interface, himrobot.rl for training, himrobot.eval for the evaluator
CLIs, and himrobot.serve for the HIM deployment adapter.
"""

# MuJoCo >= 3.12 selects its GL backend from MUJOCO_GL at `import mujoco` time, and a backend that
# is missing on the host (e.g. EGL without a driver) makes the import itself fail. Probe candidate
# backends in a subprocess and export the first one that imports cleanly; the caller's explicit
# choice is tried first. Set TT_SIM_GL_PROBE=0 to skip.
#
# This runs at `import himrobot` time (this module is imported first by every himrobot.* submodule
# and by tt_sim/__init__.py's compat shim), BEFORE any himrobot module imports mujoco itself, so the
# environment is set correctly no matter which submodule is imported first.
import os as _os
import subprocess as _subprocess
import sys as _sys


def _pick_gl_backend() -> None:
    if _os.environ.get("TT_SIM_GL_PROBE", "1") == "0" or "mujoco" in _sys.modules:
        return
    preferred = _os.environ.get("MUJOCO_GL")
    candidates = [c for c in (preferred, "egl", "osmesa", "glfw") if c]
    for backend in candidates:
        env = dict(_os.environ, MUJOCO_GL=backend)
        try:
            ok = _subprocess.run([_sys.executable, "-c", "import mujoco"], env=env, capture_output=True,
                                 timeout=60).returncode == 0
        except Exception:
            ok = False
        if ok:
            if preferred and backend != preferred:
                print(f"[himrobot] MUJOCO_GL={preferred!r} unusable; using {backend!r}", file=_sys.stderr)
            _os.environ["MUJOCO_GL"] = backend
            return
    print("[himrobot] no working MuJoCo GL backend found; leaving MUJOCO_GL as is", file=_sys.stderr)


_pick_gl_backend()
