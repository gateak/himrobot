"""predict_intercept: a pure ball-flight intercept predictor for RL use.

Thin wrapper around himrobot.policies.ik_hitter's own ball-flight forecaster
(_predict_ball_trajectory / _find_intercept / _sample_trajectory_at, plus
its PREDICTION_DELAY_S empirical bounce-timing correction) -- reused, not
reimplemented, so this predictor and the IK hitter's own plan always agree
on where/when the ball crosses the hitting plane (see
tests/test_rl_intercept.py, which checks exactly that agreement).

Unlike IKHitterPolicy._maybe_replan, this function does NOT apply the
hitter's Y_REACH/Z_REACH arm-reach clamp to the returned point -- that
clamp is an IK-workspace accommodation specific to the physical arm, not
part of the ball's predicted flight, and RL obs/reward want the raw
physics intercept so the policy can learn its own reach tradeoffs.
"""
from __future__ import annotations

import numpy as np

from himrobot.policies.ik_hitter import (
    PREDICTION_DELAY_S,
    X_HIT,
    _find_intercept,
    _predict_ball_trajectory,
    _sample_trajectory_at,
)

__all__ = ["predict_intercept"]


def predict_intercept(
    ball_pos: np.ndarray,
    ball_vel: np.ndarray,
    x_hit: float = X_HIT,
    dt: float = 0.01,
    t_max: float = 1.5,
    prediction_delay_s: float = PREDICTION_DELAY_S,
) -> tuple[np.ndarray, float, bool]:
    """Predict where/when the ball crosses the x=x_hit hitting plane.

    Forward-simulates ball_pos/ball_vel (gravity + single-bounce
    restitution, no drag -- see ik_hitter's module docstring) exactly the
    way IKHitterPolicy does, then re-samples PREDICTION_DELAY_S past the
    raw crossing to correct for the real sim's compliant (non-instant)
    bounce (see ik_hitter.PREDICTION_DELAY_S's own comment).

    Returns:
        p_int: (3,) float64 predicted intercept position. Zeros if invalid.
        t_int: float, predicted seconds from *now* until intercept. 0.0 if
            invalid.
        valid: False if the predicted trajectory never reaches x_hit within
            t_max (e.g. the ball is moving away, or falls below the table
            first) -- p_int/t_int are meaningless in that case.
    """
    traj = _predict_ball_trajectory(np.asarray(ball_pos, dtype=float),
                                     np.asarray(ball_vel, dtype=float), dt=dt, t_max=t_max)
    hit = _find_intercept(traj, x_hit)
    if hit is None:
        return np.zeros(3, dtype=np.float64), 0.0, False

    t_int, p_int, _v_int = _sample_trajectory_at(traj, hit[0] + prediction_delay_s)
    return np.asarray(p_int, dtype=np.float64), float(t_int), True
