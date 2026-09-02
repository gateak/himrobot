"""Offscreen MuJoCo -> MP4 rendering for himrobot evaluation trials.

No GUI windows: uses ``mujoco.Renderer``, which renders into an offscreen
framebuffer (EGL/OSMesa/hidden-GLFW depending on platform) and never opens
a window. The two cameras below ("diagnostic" and "broadcast") are plain
Python ``mujoco.MjvCamera`` free cameras (lookat/distance/azimuth/
elevation) rather than XML ``<camera>`` elements, so this module does not
need to touch himrobot/sim/scene.py.

Frame layout: the "broadcast" camera fills the 640x480 frame; the
"diagnostic" camera is composited as a smaller picture-in-picture inset
in the bottom-left corner, plus a translucent HUD bar across the top
(trial number, live event flags, and the running metric so far, drawn
with PIL after MuJoCo renders the frame).

Video timing: frames are captured by *sim time*, not by control step —
``TrialRecorder.maybe_capture`` is called once per control step (50 Hz)
but only actually renders+encodes a frame when enough sim time has
elapsed since the last one, so a 30 fps video plays back at (approximately)
real-time relative to the simulation, rather than at the 50 Hz control
rate (which would otherwise play back at 50/30 = 1.67x speed).
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import mujoco

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
VIDEO_FPS = 30

_INSET_WIDTH = 200
_INSET_HEIGHT = 150
_INSET_MARGIN = 8

_HUD_LINE_H = 14
_HUD_PAD = 4


def _font() -> ImageFont.ImageFont:
    # ImageFont.load_default(size=...) is only available on newer Pillow;
    # fall back to the fixed-size default bitmap font on older ones.
    try:
        return ImageFont.load_default(size=13)
    except TypeError:
        return ImageFont.load_default()


_FONT = _font()


def make_cameras() -> dict[str, "mujoco.MjvCamera"]:
    """Two fixed free (non-XML) cameras framing the table+robot scene.

    Positions are hand-picked against the layout documented in
    himrobot.sim.scene (table centred at the origin, x along the 2.74 m length,
    robot pelvis at x=-1.77 facing +x) — not derived from any cited fact.
    """
    diagnostic = mujoco.MjvCamera()
    diagnostic.type = mujoco.mjtCamera.mjCAMERA_FREE
    diagnostic.trackbodyid = -1
    diagnostic.fixedcamid = -1
    diagnostic.lookat[:] = [-0.25, 0.0, 0.95]
    diagnostic.distance = 4.6
    diagnostic.azimuth = 115  # elevated 3/4 side view showing the whole table + robot
    diagnostic.elevation = -22

    broadcast = mujoco.MjvCamera()
    broadcast.type = mujoco.mjtCamera.mjCAMERA_FREE
    broadcast.trackbodyid = -1
    broadcast.fixedcamid = -1
    broadcast.lookat[:] = [-0.15, 0.0, 0.85]
    broadcast.distance = 3.1
    broadcast.azimuth = 100  # low, near end-on TV-style angle down the table
    broadcast.elevation = -10

    return {"diagnostic": diagnostic, "broadcast": broadcast}


def _hud_lines(trial_idx: int, n_trials: int, elapsed_s: float, events: dict, metric_text: str) -> list[str]:
    flags = " ".join(
        f"{key}={'1' if events.get(key) else '0'}"
        for key in ("contact_paddle", "crossed_net", "bounce_own_side", "bounce_opponent_side", "net_hit", "out")
    )
    return [
        f"trial {trial_idx + 1}/{n_trials}   t={elapsed_s:5.2f}s",
        flags,
        metric_text,
    ]


def _draw_hud(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    img = Image.fromarray(frame).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    bar_h = _HUD_PAD * 2 + _HUD_LINE_H * len(lines)
    draw.rectangle([0, 0, img.width, bar_h], fill=(0, 0, 0, 150))
    y = _HUD_PAD
    for line in lines:
        draw.text((6, y), line, fill=(255, 255, 255, 255), font=_FONT)
        y += _HUD_LINE_H
    composited = Image.alpha_composite(img, overlay)
    return np.asarray(composited.convert("RGB"))


class TrialRecorder:
    """Renders one trial's frames (broadcast + diagnostic inset + HUD)."""

    def __init__(self, model: "mujoco.MjModel", fps: int = VIDEO_FPS):
        self._fps = fps
        self._frame_period_s = 1.0 / fps
        self._next_frame_t = 0.0
        self._cameras = make_cameras()
        self._main_renderer = mujoco.Renderer(model, height=FRAME_HEIGHT, width=FRAME_WIDTH)
        self._inset_renderer = mujoco.Renderer(model, height=_INSET_HEIGHT, width=_INSET_WIDTH)
        self.frames: list[np.ndarray] = []

    def maybe_capture(
        self,
        data: "mujoco.MjData",
        elapsed_s: float,
        trial_idx: int,
        n_trials: int,
        events: dict,
        metric_text: str,
    ) -> bool:
        """Render+store a frame iff >=1/fps sim-seconds have passed since
        the last captured frame. Returns whether a frame was captured."""
        if elapsed_s + 1e-9 < self._next_frame_t:
            return False
        self._next_frame_t += self._frame_period_s

        self._main_renderer.update_scene(data, camera=self._cameras["broadcast"])
        main = self._main_renderer.render()

        self._inset_renderer.update_scene(data, camera=self._cameras["diagnostic"])
        inset = self._inset_renderer.render()

        frame = main.copy()
        x0 = _INSET_MARGIN
        y0 = FRAME_HEIGHT - _INSET_HEIGHT - _INSET_MARGIN
        frame[y0 - 2:y0 + _INSET_HEIGHT + 2, x0 - 2:x0 + _INSET_WIDTH + 2] = 255  # white border
        frame[y0:y0 + _INSET_HEIGHT, x0:x0 + _INSET_WIDTH] = inset

        frame = _draw_hud(frame, _hud_lines(trial_idx, n_trials, elapsed_s, events, metric_text))
        self.frames.append(frame)
        return True

    def write(self, path: str) -> None:
        """Encode captured frames to an MP4 at ``path`` (H.264 via imageio/ffmpeg)."""
        import imageio.v2 as imageio

        if not self.frames:
            # Nothing was captured (e.g. a zero-length trial) — still emit a
            # single blank frame so the file exists and is a valid, playable
            # (if uninformative) video rather than a missing/corrupt output.
            self.frames.append(np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8))
        # 640x480 is already a multiple of the default macro_block_size (16),
        # so no implicit resizing happens; pixelformat defaults to yuv420p
        # (broadly-compatible H.264) already, nothing extra needed here.
        with imageio.get_writer(path, fps=self._fps, codec="libx264", quality=8) as writer:
            for frame in self.frames:
                writer.append_data(frame)

    def close(self) -> None:
        self._main_renderer.close()
        self._inset_renderer.close()


def concat_videos(paths: list[str], out_path: str, fps: int = VIDEO_FPS) -> None:
    """Concatenate several MP4s (frame-by-frame re-encode, simplest robust
    approach — avoids shelling out to ffmpeg's own concat demuxer)."""
    import imageio.v2 as imageio

    if not paths:
        return
    with imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8) as writer:
        for path in paths:
            reader = imageio.get_reader(path)
            try:
                for frame in reader:
                    writer.append_data(frame)
            finally:
                reader.close()
