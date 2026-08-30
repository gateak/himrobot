"""tt_sim.evaluate at the competition stance: pelvis welded 0.10 m toward -y.

    python -m tt_sim.evaluate_stance [every tt_sim.evaluate argument]

scene.ROBOT_Y is read from TT_ROBOT_Y once, at import, so a plain flag cannot
change it after ``python -m tt_sim.evaluate`` has started; this wrapper sets
the environment BEFORE importing anything that imports tt_sim.scene, then
defers entirely to tt_sim.evaluate. Two defaults, both overridable by
pre-setting the variable in the environment:

  TT_ROBOT_Y   -0.10   the stance measured best on the serve distribution
                       (work/tt/design/hitter_v3_reach.md section 2.1: shipped
                       policy 73.5 % -> 77.8 % legal with no retraining)
  TT_PARAM_NPZ assets/policies/param_residual_v3_s0final.npz
                       the placement-trained v3 policy (77.0 % legal, 49.5 %
                       target hits, 0.28 m landing error on seeds 11/2000/4000
                       at this stance; work/tt/rl/v3_stance/RESULT.md)

The default-stance ``tt_sim.evaluate`` and every welded-scene parity fixture
are untouched by this module's existence.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("TT_ROBOT_Y", "-0.10")
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TT_PARAM_NPZ",
                      os.path.join(_REPO, "assets", "policies", "param_residual_v3_s0final.npz"))

from tt_sim.evaluate import main  # noqa: E402  (env vars must precede the scene import)

if __name__ == "__main__":
    main(sys.argv[1:])
