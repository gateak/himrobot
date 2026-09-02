# Archive — stream alignment (August 2026)

This is the full text of `docs/PROJECT_KNOWLEDGE.md` §6b, moved here verbatim on 2026-09-01 as
part of a documentation restructure. Paths below are as they were written at the time
(`sim/align_streams.py`, `work/alignment/...`) — this is a historical record, not updated for the
later `tt_sim/` → `himrobot/`, `sim/` → `tools/mocap/` code moves. See
`docs/PROJECT_KNOWLEDGE.md` §6b for the current 5-line summary and current paths.

---

### 6b. Stream alignment (2026-08-26, Track D, Orbbec part re-run 2026-08-27 with all 8 sessions)

Script: `sim/align_streams.py` (run it with
`~/him-tools/mujoco-venv/bin/python`, which now also has scipy 1.18.1 plus
matplotlib 3.11.1). By default, it finds every `data/orbbec_*/` directory
(sorted, though `--orbbec-dir` overrides this) and records each
`data/release/orbbec_*.tar` file's byte size. Outputs are under
`work/alignment/` (git-ignored): `session-map.json`,
`imu-mocap-xcorr.csv` (all 10×8 pairs), `imu-orbbec-clock.json`,
`fbx-headers.json`, `xcorr_<episode>_<clip>.png`, `xcorr-matrix.png`,
`run.log`. `work/alignment/summarize_orbbec_clock.py` condenses the clock
JSON into `orbbec-imu-offsets.{csv,md}` (the table below). `--orbbec-motion`
adds the (inconclusive) video check. The 2026-08-27 re-run, with 8 sessions,
left `imu-mocap-xcorr.csv` the same, byte for byte, as the 2-session run
(`prev-imu-mocap-xcorr.csv`, sha256 `55555e44…`). Everything below comes from
the files. The team assumed nothing.

**Clocks.** The IMU CSV `timestamp_local` is the raw iOS wall clock, with no
zone marked. HIM's `timestamp_utc` equals local time plus 4 h
(America/New_York — stated as a *guess* in `capture-manifest.json`). The
Orbbec `session.json` and `frames.jsonl` carry epoch seconds. The session
directory name is the host wall clock at the moment the session starts.
`orbbec_20260815_214619` gives a `host_time_start` of 04:46:36.48 UTC.
`215350` gives 04:53:55.87 UTC — that is, the Orbbec host showed **UTC−7**.
Camera setup took 17.5 s and 5.9 s, between the session start and `host_time_start`.
The IMU host and the Orbbec host show the **same wall clock**.
Orbbec session 215350's frames run 21:53:56.43–21:57:09.97, and IMU episode
215358 runs 21:53:58.29–21:57:09.61. They start 1.9 s apart, and stop 0.4 s
apart, with 191.3 s of overlap. Orbbec session 214619
(21:46:36.83–21:46:42.14) falls inside IMU episode 214615
(21:46:15.93–21:46:43.50). The first frames of the other six sessions fall
−1.9…+5.4 s from their IMU episode starts (see the table below). So the "3 h"
gap between HIM's `timestamp_utc` (01:46 Z) and the Orbbec `created_utc`
(04:46 Z) is exactly UTC−4 against UTC−7. The two hosts agree on wall time,
and the Orbbec host's zone setting was UTC−7. **Two explanations both agree with the
data, and the files cannot show which one is true.** Either the IMU's
America/New_York guess is wrong by 3 h (the true UTC equals `timestamp_local`
plus 7 h). Or the Orbbec host's epoch clock was 3 h off, while both hosts
shared one zone setting. But for alignment purposes this does not matter:
align on wall clock
(`timestamp_local` against `host_time_s` minus 7 h. `imu-orbbec-clock.json`
has every offset). The remaining Orbbec-minus-IMU offsets are recorder
start/stop times, not drift (`work/alignment/orbbec-imu-offsets.md`, for the
first camera of each session. The second camera's first frame comes 0.07–0.17 s
later, and its last frame comes 0.01–0.04 s later):

| Orbbec session (tar bytes) | cam0 / cam1 serial | frames/cam | frame span s | eff. fps (prefer) | IMU episode (dur s) | first frame − IMU start s | last frame − IMU end s | span − IMU dur s | inside IMU |
|---|---|---|---|---|---|---|---|---|---|
| 214619 (59,699,200) | …8E / …8F | 125 | 5.315 | 23.5 (30) | 214615 (27.571) | +20.897 | −1.359 | −22.256 | yes |
| 214831 (3,175,403,520) | …8F / …8E | 6,578 | 260.330 | 25.3 (30) | 214836 (257.341) | −1.901 | +1.088 | +2.989 | no (starts 1.9 s early) |
| 215350 (1,910,609,920) | …8E / …8F | 3,983 | 193.540 | 20.6 (30) | 215358 (191.311) | −1.866 | +0.363 | +2.229 | no (starts 1.9 s early) |
| 215750 (3,139,819,520) | …8F / …8E | 6,554 | 325.966 | 20.1 (30) | 215749 (335.283) | +5.394 | −3.923 | −9.317 | yes |
| 220448_60fps (2,738,247,680) | …8E / …8F | 5,673 | 297.057 | 19.1 (60) | 220452 (301.526) | +3.779 | −0.690 | −4.469 | yes |
| 221035_60fps (2,808,535,040) | …8F / …8E | 5,803 | 301.837 | 19.2 (60) | 221043 (300.932) | +0.456 | +1.361 | +0.905 | no (ends 1.4 s late) |
| 221649_60fps (2,871,398,400) | …8E / …8F | 5,927 | 309.195 | 19.2 (60) | 221654 (303.722) | +0.776 | +6.249 | +5.473 | no (ends 6.2 s late) |
| 222542_60fps (2,813,112,320) | …8F / …8E | 5,773 / 5,772 | 306.769 | 18.8 (60) | 222546 (311.222) | +4.249 | −0.204 | −4.453 | yes |

Wall-clock first frames: 214831 21:48:34.43, 215750 21:57:54.93, 220448
22:04:56.35, 221035 22:10:43.72, 221649 22:16:55.35, 222542 22:25:50.76 (all
on the UTC−7 host wall clock. `implied_host_utc_offset_h` is −7 for every
session. The gap from session start to `host_time_start` is 2.9–8.3 s, except
17.5 s for 214619). The `_60fps` sessions' `frames.jsonl` spans agree with
their IMU episode durations to within −4.5…+5.5 s. The first/last-frame
offsets explain the whole-session mismatch — for example, session 221649
started 0.8 s after the IMU episode, and stopped 6.2 s after it. The data
shows no clock drift at this resolution. **The target `fps_prefer: 60` was
not reached**: the four `_60fps` sessions have a median frame dt of
0.050–0.051 s (18.8–19.2 fps, effective. 5,673–5,927 frames, over about 300
s), and the 30 fps sessions have a median frame dt of 0.037–0.050 s
(20.1–25.3 fps). `session.json`'s `frame_counts` equal the `frames.jsonl` row
counts for the 7 sessions that have this field (215350 stopped early and has
none). Camera labels follow USB enumeration order, **not** the serial number.
Sessions 214831, 215750, 221035_60fps, and 222542_60fps have
`cam0_CP0E8530008F` and `cam1_CP0E8530008E` — the reverse of the other four.
So identify cameras by serial number, not by `cam0`/`cam1`. An independent
check used frame-difference motion energy from the Orbbec color stream
against |gyro|. This check gives no clear answer. For 215350, r ≈ 0.10 at the expected
lag, with no peak above z 2.7 in ±60 s, on both cameras. The whole-frame
signal is dominated by the other player and the ball. The team repeated this
on 2026-08-27, on session 220448_60fps (all 5,673 color JPGs per camera, with
no frame subsampling. Results in `work/alignment/motion60/`): camera 1 (…8F) has its
best lag at **+4.10 s, against an expected +4.13 s** (r 0.195, z 2.8,
secondary ratio 0.83). Camera 0 (…8E) shows r −0.23 at the expected +3.99 s,
with a false peak at +58.0 s (r 0.18, z 2.5). Camera 1's result supports the
shared-wall-clock rule. But neither camera clears the script's own bar for a
clear answer (z ≥ 4, secondary ratio < 0.6). So the result stays
inconclusive.

**Rokoko FBX headers** (binary FBX 7500, parsed node by node).
`CreationTimeStamp` runs 2026-08-17 13:08:23 (Clip) through 13:31:04
(Clip_8) — that is, export time, two days after capture. This is in the export
host's own, unstated time zone. `Original|DateTime_GMT` and
`LastSaved|DateTime_GMT` are empty. One take is named `clip` (`clip.tak`).
Take length equals clip length (17.43, 252.87, 192.90, 318.93, 295.37,
298.60, 305.23, 307.90 s). `TimeMode` is 6, meaning 30 fps. The data is Y-up,
in cm. Clip_8 was exported from a Rokoko folder named `Clip (1)`. **No
recording timestamp exists in the FBX file**. The team anchors Rokoko only
through the IMU.

**IMU-to-Rokoko method.** Frame 0 of every Blender BVH file is a T-pose (all
rotation channels are 0). The team drops it, so **clip time t = 0 is BVH
frame 1**. Every lag below uses this origin. The BVH FK uses standard
semantics, with the 6-channel position values *replacing* the OFFSET value
(the Blender exporter writes rest-offset plus translation). The team checked
this against Blender's own `pose_bone.head`, on Clip.fbx (exact for Hips,
mean 1.2 cm and max 2.7 cm for RightHand). The signal is the RightHand
**world angular-speed magnitude** (from the FK rotation matrices), against
IMU |gyro|. This is the same physical quantity, for a paddle held rigidly (p99: hand
520–615 dps, IMU 670–740 dps). The team kept linear hand speed as the
`alt_feature` (this gives a lower r everywhere). The team resampled both signals
to 10 Hz, with a Gaussian σ of 0.4 s. The team z-scored and clipped them at ±4σ. The team
took the Pearson r over the overlap, for lags of ±60 s (with a minimum 15 s
overlap). Lag sign: clip time t lines up with IMU episode time t + lag. A
negative lag means the IMU recording started after the clip.
`secondary_ratio` is the best local maximum more than 5 s away, divided by
the peak. `lobe` is the lag span where r ≥ 0.8 of the peak. `peak-match` is
the most common value among pairwise swing-peak differences (peaks over 400
dps, σ 0.1 s).

| IMU episode (wall, s) | Orbbec session | Rokoko clip (s) | lag s | r | 2nd ratio | z | lobe s | peak-match lag | runner-up clip (r) | confidence |
|---|---|---|---|---|---|---|---|---|---|---|
| 214615 (21:46:15, 27.6) | 214619 (on disk, 5.3 s inside, +20.9 s) | Clip (17.5) | +6.8 | 0.987 | 0.39 | 2.3 | 1.1 | — (no peaks >400 dps) | — | test take: four slow paddle wobbles match exactly, but only ~17 s of signal |
| 214836 (21:48:36, 257.3) | 214831 (on disk, first frame −1.9 s) | Clip_2 (252.9) | −3.8 | 0.636 | 0.32 | 4.3 | 2.1 | −4.2 | Clip_5 (0.244) | high |
| 215358 (21:53:58, 191.3) | 215350 (on disk, 193.5 s, −1.9 s) | Clip_3 (192.9) | +0.9 | 0.484 | 0.46 | 3.6 | 1.7 | −14.3 (disagrees) | none by duration | medium |
| 215749 (21:57:49, 335.3) | 215750 (on disk, +5.4 s) | Clip_4 (319.0) | −1.2 | 0.446 | 0.41 | 3.9 | 2.0 | −1.3 | Clip_7 (0.223) | medium |
| 220452 (22:04:52, 301.5) | 220448_60fps (on disk, +3.8 s) | Clip_5 (295.4) | +0.3 | 0.452 | 0.47 | 3.5 | 2.7 | −19.4 (disagrees) | Clip_7 (0.233) | medium |
| 221043 (22:10:43, 300.9) | 221035_60fps (on disk, +0.5 s) | Clip_6 (298.6) | −2.7 | 0.364 | 0.81 | 3.0 | 3.0 | −28.9 (disagrees) | Clip_4 (0.201) | medium (weakest) |
| 221654 (22:16:54, 303.7) | 221649_60fps (on disk, +0.8 s) | Clip_7 (305.3) | −4.8 (twin peak −1.6, r 0.39, linear-speed feature picks −1.6) | 0.422 | 0.51 | 2.7 | 4.2 | −1.8 | Clip_5 (0.196) | low (lag ambiguous between −4.8 and −1.6) |
| 222546 (22:25:46, 311.2) | 222542_60fps (on disk, +4.2 s) | Clip_8 (307.9) | −6.0 | 0.624 | 0.46 | 4.3 | 2.7 | −6.2 | Clip_5 (0.257) | high |
| 192602 (19:26:02, 21.7) | — | — (Clip is taken by 214615, r 0.58 here but IMU peaks 2087 dps vs 271 dps in the clip) | | | | | | | | none |
| 215351 (21:53:51, 3.8) | — (false start before 215358) | — | | | | | | | | none |

The four sessions of about 300 s each (220452/221043/221654/222546, against
Clip_5…8) cannot be told apart by duration alone. But the 4×4 correlation
block is diagonal, both by row and by column (diagonal r 0.36–0.62, every
off-diagonal value ≤ 0.28). So the chronological pairing — IMU to clip, in
recording order — holds. The implied start/stop consistency is IMU end minus
clip end, equal to IMU duration minus lag minus clip duration. The values are:
  - Clip_2 +8.2 s.
  - Clip_3 −2.5 s.
  - Clip_4 +17.5 s.
  - Clip_5 +5.8 s.
  - Clip_6 +5.0 s.
  - Clip_7 +3.2 s (0.0 s at the −1.6 s lag).
  - Clip_8 +9.3 s.
  - Clip +3.3 s.

That is, the
IMU usually started a few seconds after the Rokoko take, and stopped a few
seconds after it too. This agrees with one operator working two apps. Chain
for the one fully tripled session: Clip_3 frame 0 equals IMU 215358 t = 0.9
s, equal to wall time 21:53:59.2. Orbbec 215350 frame 0 equals wall time
21:53:56.43. So `clip3_t = orbbec_host_time_s − 7 h − 21:53:59.2`.

**Caveats.** The IMU runs at a nominal 11.11 Hz, but effectively at **about
10.1 Hz** (samples divided by duration, in `work/alignment/run.log`). For
example, 1,932 samples over 191.3 s for 215358, and 3,384 samples over 335.3
s for 215749. So swings, which last about 0.2 s, are under-sampled: r stays
moderate (0.36–0.64), and the lag lobes are 1.7–4.2 s wide. Treat every lag
as accurate to only ±0.5 s, and refine it per swing if this is necessary.
Peak matching finds about twice as many IMU peaks as hand peaks, and it
agrees with the cross-correlation result only for Clip_2, Clip_4, and Clip_8.
The Orbbec-motion check neither confirms nor contradicts the shared-wall-clock
idea. All eight Orbbec sessions are now on disk (tar files 59.7 MB to 3.18
GB, extracted under `data/`). So IMU-to-Orbbec offsets exist for every one of
them. But these offsets rest on the shared-wall-clock idea, accurate to
about ±0.1 s, from frame timing. The two cameras of one session start
0.07–0.17 s apart. The 214619 `frames.jsonl` span (5.315 s) is shorter than
its `session.json` `duration_s` (5.885 s). This is because the second value runs from
`host_time_start_s` to `host_time_end_s`, not from the first frame to the
last frame. Rokoko carries no absolute clock. The team used a 0.1 s lag grid.

