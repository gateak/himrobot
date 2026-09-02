"""``SwingLibraryCommand`` -- the Isaac-side stand-in for the shipped hitter
during WBC training: a pre-recorded swing library played back per env
instead of a live IK solve. docs/design/whole_body_v1.md sec 4.1 (the
libraries), sec 6.1 (the curriculum: library mix by iteration, reset-source
mix, adaptive row sampling). See the ``mdp`` package docstring for the
three-tier split this file follows -- the isaaclab ``CommandTerm``/
``CommandTermCfg`` subclass that wires the functions below into a real Isaac
Lab CommandManager lives in ``env_cfg.py``.

Library npz schema -- ``himrobot.wbc.replay_library.LIBRARY_KEYS`` (WB4,
landed; WB1's ``himrobot/wbc/record_library.py``, the writer, has not landed
yet -- this module reads the schema WB4 already committed to):

    u                float32 [R, T, 17]  JOINT_ORDER[12:29] targets (waist
                                          yaw/roll/pitch + left arm x7 +
                                          right arm x7), one per control step
    swing_start_step int32   [R]         first step where u differs from
                                          scene.STAND_POSE[12:]
    contact_step     int32   [R]         control step nearest predicted contact
    t_contact        float32 [R]         seconds (row's own episode clock)
    t_pre            float32 [R]         seconds, backswing/ease-in width
    t_follow         float32 [R]         seconds, follow-through width
    serve_pos        float32 [R, 3]      the row's serve (unused on the
    serve_vel        float32 [R, 3]      ball-free Isaac task; kept for
                                          logging/replay-parity only)
    welded_legal     bool    [R]         legal_return()/contact when this
    welded_contact   bool    [R]         row was recorded on the welded scene

``serve_scale`` (the design doc's critic-only observation slot) is NOT a
per-row npz field (WB4's schema has none) -- it is a per-LIBRARY scalar
(nominal 1.0, wide 1.3, slow 1.0 -- docs/design/whole_body_v1.md sec 4.1)
supplied by the caller (``SwingLibrary.from_npz(path, serve_scale=...)``)
and broadcast across that library's rows.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from himrobot.sim import scene

N_LEG = 12
N_WAIST = 3
N_ARM = 14
N_CMD = N_WAIST + N_ARM  # 17, matches mdp.observations.N_CMD / mdp.actions' u slice
assert N_LEG + N_CMD == scene.NUM_JOINTS

# STAND_POSE[12:29]: what an idle row commands on every step, and what a
# library row's u[t] is measured against (swing_start_step's definition).
STAND_CMD = torch.as_tensor(scene.STAND_POSE[N_LEG:], dtype=torch.float32)  # (17,)

PREVIEW_OFFSETS = (0, 5, 10, 15, 25)  # u(t+k) horizons this command exposes (steps @ 50 Hz)
TIME_JITTER_STEPS = 3  # library time-jitter, sec 6.1: "not perfectly phase-locked"
POST_SWING_CAPTURE_OFFSET = 25  # contact_step + 25, HOPE's own-state ring-buffer mechanism, sec 6.1
RING_BUFFER_SIZE = 8192
RING_MIN_FILL = 256  # below this, "post_swing_ring" reset mass folds into "pre_swing" (own choice, not in the design doc)
ADAPTIVE_EMA_ALPHA = 0.001
ADAPTIVE_FLOOR = 0.1
SPEED_BUFFER = 0.0  # himrobot.wbc.replay_library._fake_plan's own documented approximation (offline, no live plan)

# Reset-source mix (sec 6.1): row step 0 / mid-swing with pose+joint noise /
# the policy's own post-swing ring buffer / idle (STAND_POSE throughout).
RESET_MIX = {"row_start": 0.25, "pre_swing": 0.45, "post_swing_ring": 0.25, "idle": 0.05}
assert abs(sum(RESET_MIX.values()) - 1.0) < 1e-9

# Library mix by training iteration (sec 6.1): the fraction of resets drawn
# from each label, in force from the given iteration onward.
LIBRARY_MIX_SCHEDULE: list[tuple[int, dict[str, float]]] = [
    (0, {"slow": 0.70, "nominal": 0.30, "wide": 0.00}),
    (1000, {"slow": 0.20, "nominal": 0.60, "wide": 0.20}),
    (3000, {"slow": 0.10, "nominal": 0.50, "wide": 0.40}),
]
for _threshold, _mix in LIBRARY_MIX_SCHEDULE:
    assert abs(sum(_mix.values()) - 1.0) < 1e-9, _mix


def library_mix_at(iteration: int) -> dict[str, float]:
    """The label -> fraction dict in force at ``iteration`` -- the last
    schedule entry whose threshold is <= iteration."""
    mix = LIBRARY_MIX_SCHEDULE[0][1]
    for threshold, m in LIBRARY_MIX_SCHEDULE:
        if iteration >= threshold:
            mix = m
    return mix


def restrict_mix(mix: dict[str, float], loaded_labels: list[str]) -> dict[str, float]:
    """``mix`` reduced to the labels a run actually loaded and renormalised
    to sum to 1. A run that loads none of the schedule's labels (Gate 0's
    single ``eval`` library, a debugging run with one file) falls back to
    uniform over what it has -- without this, sampling asked the bank for a
    label it never held and raised."""
    kept = {label: w for label, w in mix.items() if label in loaded_labels and w > 0.0}
    total = sum(kept.values())
    if total <= 0.0:
        return {label: 1.0 / len(loaded_labels) for label in loaded_labels}
    return {label: w / total for label, w in kept.items()}


def pad29(u17: torch.Tensor) -> torch.Tensor:
    """(..., 17) JOINT_ORDER[12:29] targets -> (..., 29) with zeros on the leg
    slots -- himrobot.wbc.replay_library._pad29's torch twin. Both
    mdp.observations.compute_actor_obs and mdp.actions.joint_target_wbc
    read only [12:29] of a 29-dim ``u`` (legs come from leg_default), so the
    padding is never read; it exists so the library's 17-wide rows fit the
    29-wide contract those functions share with the numpy runtime."""
    zeros = torch.zeros(u17.shape[:-1] + (N_LEG,), dtype=u17.dtype, device=u17.device)
    return torch.cat([zeros, u17], dim=-1)


def apply_idle(u17: torch.Tensor, idle: torch.Tensor) -> torch.Tensor:
    """Rows flagged ``idle`` (N,) bool read STAND_POSE[12:] instead of their
    library frame -- the design's 5 % idle resets (sec 6.1: "u == STAND_POSE
    for 150 steps") without needing a dedicated all-STAND row in the bank."""
    stand = STAND_CMD.to(device=u17.device, dtype=u17.dtype)
    return torch.where(idle.unsqueeze(-1), stand.expand_as(u17), u17)


# ---------------------------------------------------------------------------
# pure torch: library storage + the gather/clock math tests exercise directly.
# ---------------------------------------------------------------------------
@dataclass
class SwingLibrary:
    """One loaded library as GPU (or CPU, for tests) tensors, plus its own
    per-row EMA failure weight (adaptive sampling, sec 6.1) and a per-library
    ``serve_scale`` broadcast across every row (see module docstring)."""

    label: str
    u: torch.Tensor                 # (R, T, 17)
    swing_start_step: torch.Tensor  # (R,) int64
    contact_step: torch.Tensor      # (R,) int64
    t_contact: torch.Tensor         # (R,) float32, seconds
    t_pre: torch.Tensor             # (R,) float32
    t_follow: torch.Tensor          # (R,) float32
    serve_scale: float = 1.0
    fail_ema: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        assert self.u.dim() == 3 and self.u.shape[-1] == N_CMD, self.u.shape
        for name in ("swing_start_step", "contact_step", "t_contact", "t_pre", "t_follow"):
            assert getattr(self, name).shape == (self.n_rows,), name
        self.fail_ema = torch.full((self.n_rows,), ADAPTIVE_FLOOR, dtype=torch.float32, device=self.u.device)

    @property
    def n_rows(self) -> int:
        return int(self.u.shape[0])

    @property
    def n_steps(self) -> int:
        return int(self.u.shape[1])

    @classmethod
    def from_npz(cls, path: str, label: str | None = None, serve_scale: float = 1.0,
                device: torch.device | str = "cpu") -> "SwingLibrary":
        """Loads himrobot.wbc.replay_library.LIBRARY_KEYS' ``u``,
        ``swing_start_step``, ``contact_step``, ``t_contact``, ``t_pre``,
        ``t_follow`` (``serve_pos``/``serve_vel``/``welded_legal``/
        ``welded_contact`` are not needed on the ball-free Isaac task and
        are not loaded here)."""
        z = np.load(path, allow_pickle=False)
        u = torch.as_tensor(np.asarray(z["u"], dtype=np.float32), device=device)
        return cls(
            label=label or path,
            u=u,
            swing_start_step=torch.as_tensor(np.asarray(z["swing_start_step"], dtype=np.int64), device=device),
            contact_step=torch.as_tensor(np.asarray(z["contact_step"], dtype=np.int64), device=device),
            t_contact=torch.as_tensor(np.asarray(z["t_contact"], dtype=np.float32), device=device),
            t_pre=torch.as_tensor(np.asarray(z["t_pre"], dtype=np.float32), device=device),
            t_follow=torch.as_tensor(np.asarray(z["t_follow"], dtype=np.float32), device=device),
            serve_scale=float(serve_scale),
        )

    def record_outcomes(self, row_idx: torch.Tensor, failed: torch.Tensor) -> None:
        """EMA-update the failure weight of every row in ``row_idx``
        (duplicates accumulate one update each, in index order -- fine for
        an EMA). Floor keeps every row reachable by sample_rows even after
        a long streak of successes."""
        target = failed.to(self.fail_ema.dtype)
        self.fail_ema[row_idx] = (1.0 - ADAPTIVE_EMA_ALPHA) * self.fail_ema[row_idx] + ADAPTIVE_EMA_ALPHA * target
        self.fail_ema.clamp_(min=ADAPTIVE_FLOOR)

    def sample_rows(self, n: int, generator: torch.Generator | None = None) -> torch.Tensor:
        """n row indices, drawn with replacement, weighted by this
        library's own failure EMA."""
        return torch.multinomial(self.fail_ema, n, replacement=True, generator=generator)


def gather_u(u: torch.Tensor, row_idx: torch.Tensor, time_idx: torch.Tensor) -> torch.Tensor:
    """u: (R, T, C). row_idx/time_idx: (N,) int64, same length. time_idx is
    clamped to [0, T-1] -- a request past a row's recorded length repeats
    its last frame (the row's own post-swing hold) rather than indexing out
    of bounds."""
    assert row_idx.shape == time_idx.shape
    t = torch.clamp(time_idx, 0, u.shape[1] - 1)
    return u[row_idx, t]


def gather_previews(u: torch.Tensor, row_idx: torch.Tensor, time_idx: torch.Tensor,
                    offsets: tuple[int, ...] = PREVIEW_OFFSETS) -> dict[int, torch.Tensor]:
    """{offset: (N, C)} for every offset in ``offsets`` -- u(t), u(t+5), ...
    each via gather_u's own clamp rule."""
    return {k: gather_u(u, row_idx, time_idx + k) for k in offsets}


def gather_scalar(values: torch.Tensor, row_idx: torch.Tensor) -> torch.Tensor:
    """values: (R,). row_idx: (N,) int64. Per-row scalar fields
    (t_contact/t_pre/t_follow/...) gathered the same way gather_u gathers
    the (T, C) rows -- a plain advanced-index, no time dimension to clamp."""
    return values[row_idx]


@dataclass
class LibraryBank:
    """Every loaded label ("nominal"/"wide"/"slow"/...) concatenated into
    ONE set of (R_total, ...) tensors, so a per-env step needs exactly one
    gather regardless of how many labels an env's row might be drawn from
    (docs/design/whole_body_v1.md sec 4.1: "the library is one GPU tensor,
    one gather per step"). Shorter ``u`` rows are padded by repeating their
    own last frame (consistent with gather_u's own clamp-to-last-frame
    rule, so padding never changes what a step reads); the per-row scalar
    fields need no padding."""

    labels: list[str]
    row_offset: dict[str, int]     # label -> its rows' starting index in the combined tensors
    u: torch.Tensor                # (R_total, T_max, 17)
    swing_start_step: torch.Tensor  # (R_total,)
    contact_step: torch.Tensor      # (R_total,)
    t_contact: torch.Tensor         # (R_total,)
    t_pre: torch.Tensor             # (R_total,)
    t_follow: torch.Tensor          # (R_total,)
    serve_scale: torch.Tensor       # (R_total,) -- each library's scalar, broadcast per row
    fail_ema: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.fail_ema = torch.full((self.n_rows,), ADAPTIVE_FLOOR, dtype=torch.float32, device=self.u.device)

    @property
    def n_rows(self) -> int:
        return int(self.u.shape[0])

    @property
    def n_steps(self) -> int:
        return int(self.u.shape[1])

    @classmethod
    def build(cls, libraries: dict[str, SwingLibrary]) -> "LibraryBank":
        assert libraries, "at least one library required"
        labels = list(libraries.keys())
        t_max = max(lib.n_steps for lib in libraries.values())
        device = next(iter(libraries.values())).u.device

        us, starts, contacts, t_contacts, t_pres, t_follows, scales, offset, offsets = (
            [], [], [], [], [], [], [], 0, {}
        )
        for label in labels:
            lib = libraries[label]
            pad = t_max - lib.n_steps
            u = lib.u if pad == 0 else torch.cat([lib.u, lib.u[:, -1:, :].expand(-1, pad, -1)], dim=1)
            us.append(u)
            starts.append(lib.swing_start_step)
            contacts.append(lib.contact_step)
            t_contacts.append(lib.t_contact)
            t_pres.append(lib.t_pre)
            t_follows.append(lib.t_follow)
            scales.append(torch.full((lib.n_rows,), lib.serve_scale, dtype=torch.float32, device=device))
            offsets[label] = offset
            offset += lib.n_rows

        return cls(
            labels=labels, row_offset=offsets,
            u=torch.cat(us, dim=0),
            swing_start_step=torch.cat(starts, dim=0),
            contact_step=torch.cat(contacts, dim=0),
            t_contact=torch.cat(t_contacts, dim=0),
            t_pre=torch.cat(t_pres, dim=0),
            t_follow=torch.cat(t_follows, dim=0),
            serve_scale=torch.cat(scales, dim=0),
        )

    def _n_rows_in(self, label: str) -> int:
        idx = self.labels.index(label)
        start = self.row_offset[label]
        end = self.row_offset[self.labels[idx + 1]] if idx + 1 < len(self.labels) else self.n_rows
        return end - start

    def sample_rows(self, label: str, n: int, generator: torch.Generator | None = None) -> torch.Tensor:
        """n GLOBAL (combined-tensor) row indices, drawn from ``label`` only,
        weighted by that label's own failure EMA."""
        start, count = self.row_offset[label], self._n_rows_in(label)
        weights = self.fail_ema[start:start + count]
        return torch.multinomial(weights, n, replacement=True, generator=generator) + start

    def record_outcomes(self, row_idx: torch.Tensor, failed: torch.Tensor) -> None:
        """Global-index EMA update -- see SwingLibrary.record_outcomes."""
        target = failed.to(self.fail_ema.dtype)
        self.fail_ema[row_idx] = (1.0 - ADAPTIVE_EMA_ALPHA) * self.fail_ema[row_idx] + ADAPTIVE_EMA_ALPHA * target
        self.fail_ema.clamp_(min=ADAPTIVE_FLOOR)


@dataclass
class PostSwingRing:
    """A fixed-capacity circular buffer of the policy's own captured
    post-swing states (sec 6.1: "the falls land 1.8-2.9s into the episode,
    in the recovery phase, so that phase needs its own samples"). Each
    pushed state is caller-defined (a dict of tensors); this class only
    manages the circular indexing and uniform sampling."""

    capacity: int = RING_BUFFER_SIZE

    def __post_init__(self) -> None:
        self._states: list[dict] = []
        self._next = 0

    @property
    def filled(self) -> int:
        return len(self._states)

    def push(self, state: dict) -> None:
        if len(self._states) < self.capacity:
            self._states.append(state)
        else:
            self._states[self._next] = state
            self._next = (self._next + 1) % self.capacity

    def sample(self, n: int, generator: torch.Generator | None = None) -> list[dict]:
        if not self._states:
            return []
        idx = torch.randint(0, len(self._states), (n,), generator=generator)
        return [self._states[int(i)] for i in idx.tolist()]


def sample_reset_sources(n: int, ring_filled: int, generator: torch.Generator | None = None,
                         min_fill: int = RING_MIN_FILL) -> list[str]:
    """n reset-source labels drawn per RESET_MIX -- "post_swing_ring"'s share
    folds into "pre_swing" until the ring holds at least ``min_fill``
    states, so early training never samples an empty ring."""
    mix = dict(RESET_MIX)
    if ring_filled < min_fill:
        mix["pre_swing"] += mix.pop("post_swing_ring")
        mix["post_swing_ring"] = 0.0
    labels = list(mix.keys())
    weights = torch.tensor([mix[label] for label in labels], dtype=torch.float32)
    idx = torch.multinomial(weights, n, replacement=True, generator=generator)
    return [labels[int(i)] for i in idx.tolist()]
