# Archive — perception status (August 2026)

This is the full text of `docs/PROJECT_KNOWLEDGE.md` §6a, moved here verbatim on 2026-09-01 as
part of a documentation restructure (the section was about half the document and specific to one
week of perception work). Paths below are as they were written at the time (`sim/...`,
`work/perception/...`) — this is a historical record, not updated for the later `tt_sim/` →
`himrobot/`, `sim/` → `tools/mocap/` code moves. See `docs/PROJECT_KNOWLEDGE.md` §6a for the
current 5-line summary and current paths.

---

### 6a. Perception status (2026-08-26 Track C, updated 2026-08-27 Track P)

**What is on disk (2026-08-27).** All eight Orbbec sessions are now real directories, `data/orbbec_20260815_*/`, taken from the ten verified tar files in `data/release/` (`data/release/extract.log`, extracted 2026-08-27). None of them is a 0-byte download any more. Per camera (`work/perception/sessions.json`):
  - `214619` — 125 rows, 5.3 s, 23.3–24.0 fps host (**no ball — a player adjusts a paddle**).
  - `215350` — 3,983 rows, 193.5 s, 20.6 fps (no `host_time_end` and no `frame_counts` — the recorder stopped early. It carries serves and rallies from about frame 100).
  - `214831` — 6,578 rows, 260.3 s, 25.3 fps.
  - `215750` — 6,554 rows, 326.0 s, 20.1 fps.
  - `220448_60fps` — 5,673 rows, 297.1 s, 19.1 fps.
  - `221035_60fps` — 5,803 rows, 301.8 s, 19.2 fps.
  - `221649_60fps` — 5,927 rows, 309.2 s, 19.2 fps.
  - `222542_60fps` — 5,773/5,772 rows, 306.8 s, 18.8 fps.

**The `_60fps` sessions are not 60 fps on disk.** The cameras did run at 59.9 Hz. The color `device_timestamp_us` steps 50.09 ms while the device frame `index` steps by 3. That is, 16.7 ms per device frame (`work/perception/device_index_rate.json`). But the recorder kept only about 1 of every 3.1 device frames (host `dt` median 50.3–51.0 ms, p95 61–64 ms, max 0.33–0.56 s). So the effective rate is 18.8–19.2 fps. This is *lower* than the 30 fps sessions:
  - 214831 kept 1 of every 1.18 frames, giving 25.3 fps.
  - 215350/215750 kept 1 of every 1.45–1.49 frames, giving 20.1–20.6 fps.

The `_60fps` sessions give no time advantage. The ball still jumps 50–80 px between recorded frames. **The camera labels do not point to the same serial number across sessions.** Sessions 214619/215350/220448/221649 have `cam0_CP0E8530008E` plus `cam1_CP0E8530008F`. Sessions 214831/215750/221035/222542 have `cam0_CP0E8530008F` plus `cam1_CP0E8530008E`. This depends on USB enumeration order. So identify each camera only by its serial number, never by the cam index. The physical cameras never moved. Serial …8E always sees ArUco IDs 4–7, and …8F always sees IDs 0–3. Marker centers stay within 3.7 px of 215350's centers in all eight sessions (`work/perception/aruco_compare.md`). Null rows are rare (0–9 for color, 0–5 for depth, per camera. `idx` stays continuous everywhere) — the recorder did not record that stream for that row, and the detector does not use these rows. Frame 10 of camera 1 in session 214619 has damage in the lower third of the picture.

**Camera geometry (from the frames, not from the metadata).** The two cameras are *not* a classic overlapping stereo pair. cam0 (`CP0E8530008E`) looks down one half of the table, with the net at its left image edge. cam1 (`CP0E8530008F`) looks down the other half, with the net at its right image edge. The only overlap is the net strip. ArUco: `DICT_4X4_50` decodes. Its 100/250/1000 supersets also decode. No 5×5, 6×6, 7×7, or AprilTag marker decodes. There are **4 markers at each half's four corners**:
  - cam0 sees IDs 4 (near/net, about (86,425)), 5 (far/net, about (190,101)), 6 (near/end-line, about (567,427)), 7 (far/end-line, about (472,86)).
  - cam1 sees IDs 0 (near/end, about (87,408)), 1 (far/end, about (203,97)), 2 (near/net, about (564,435), partly behind the net post), 3 (far/net, about (481,103)).

Over 12 sampled frames (`work/perception/aruco_orbbec_20260815_215350.json`, `n_frames`), the decode counts are: cam0 IDs 4 → 12, 6 → 12, 5 → 9, 7 → 3. cam1 IDs 0 → 12, 3 → 8, 1 → 7, 2 → 3. The far markers, at about 20 px, decode in 3 to 9 of 12 frames. The near markers decode in 12 of 12, except ID 2 (at the net post), which decodes in 3 of 12. The median pixel corners for each ID are in `work/perception/aruco_orbbec_20260815_*.json`. The claim in §6, that markers sit at the table corners, is **confirmed** per half. The markers on the far half are visible across the net, but the detector does not decode them. The marker's physical size, and its exact offset from the table corner, are **not yet known** — both values are necessary for solvePnP.

**2D detector (`sim/ball_detect.py`).** This uses an HSV orange mask plus roundness. It gates candidates with:
  - Frame differencing (`--motion 8`).
  - The ArUco-hull table ROI (`--roi-json`, `--roi-margin`).
  - A static-orange map (`--static-reject 0.2`), which removes the bottle by the stool and a small orange object on the floor near the far player.
  - Distance from large red or orange blobs (`--near-big 12`, the paddle rubber).

It reads a directory or the tar file, with `--cam cam0|cam1`. The team measured ball color in OpenCV HSV: the median over a disc of 0.6× the detected radius, around the `ball2d` center. This was for session 215350 cam0, checked again on 2026-08-27 with `~/him-tools/cv-venv`. Two sets of hits show:
  - Six confirmed ball hits (frames 255/1799/2518/2954/3058/3638): H 18–20, S 159–194, V 213–234.
  - Four sandal hits (frames 2067/2246/2406/2623): H 16–17, S 112–125, V 99–145.

The hue values overlap. Over the full detected disc, the motion-blur at the ball's edge pulls its median S down to 97–122 — into the sandal range. So color alone cannot tell the two apart. Rates over the full sessions (gated, ROI margin 40 px):

| session / cam | frames | candidates | sampled precision |
|---|---|---|---|
| 215350 cam0 | 3,980 | 1,270 (31.9 %) | 8/12 real ball (4 = far player's sandals) |
| 215350 cam1 | 3,978 | 1,043 (26.2 %) | 5/12 real ball (7 = sandals/feet at the far edge) |
| 214619 cam0 | 123 | 20 (16.3 %) | 0/20 — all the far player's sandal |
| 214619 cam1 | 121 | 0 | — |

The team ran six new sessions on 2026-08-27 (Track P), with the same gated
settings and `--step 1` (every frame). Each session has its own ROI file
(`work/perception/roi_orbbec_20260815_<s>.json`). The 12-frame ArUco sample
misses ID 2 on serial …8F in sessions 214831 and 221035, and misses ID 1 in
222542. For those, the team filled the corners in from 215350's medians for the
same serial. The driver is `work/perception/run_track_p.sh`, 35–43 s per session per
camera. All numbers are in `work/perception/summary.md`, `summary.json`, and
`precision_counts.json`. Precision here means 12 random candidates per
camera, seed 0, checked by eye on
`precision_check_<s>_<cam>_random12.jpg`. "Depth resolved" means the disc
median is more than 40 mm nearer than the annulus around it:

| session / cam (serial) | frames | candidates | depth resolved | random-12 real / FP (FP kinds) |
|---|---|---|---|---|
| 214831 cam0 (…8F) | 6,574 | 1,506 (22.9 %) | 881 (59 %) | 9 / 3 (sandal 2, foot 1) |
| 214831 cam1 (…8E) | 6,572 | 1,608 (24.5 %) | 1,102 (69 %) | 8 / 4 (sandal 3, hand 1) |
| 215750 cam0 (…8F) | 6,545 | 1,896 (29.0 %) | 1,221 (64 %) | 10 / 2 (sandal 1, hand 1) |
| 215750 cam1 (…8E) | 6,552 | 2,796 (42.7 %) | 1,587 (57 %) | 5 / 7 (sandal 7 — one standing far player's sandal at ≈(192–206, 80) px) |
| 220448_60fps cam0 (…8E) | 5,672 | 2,232 (39.4 %) | 1,029 (46 %) | 6 / 6 (sandal 6) |
| 220448_60fps cam1 (…8F) | 5,672 | 1,476 (26.0 %) | 970 (66 %) | 12 / 0 |
| 221035_60fps cam0 (…8F) | 5,803 | 2,117 (36.5 %) | 1,508 (71 %) | 10 / 2 (hand/paddle 1, sandal 1) |
| 221035_60fps cam1 (…8E) | 5,803 | 1,893 (32.6 %) | 1,075 (57 %) | 9 / 3 (sandal 3) |
| 221649_60fps cam0 (…8E) | 5,925 | 2,003 (33.8 %) | 1,304 (65 %) | 6 / 6 (wrist above a black sleeve 5, sandal 1) |
| 221649_60fps cam1 (…8F) | 5,926 | 2,793 (47.1 %) | 1,259 (45 %) | 6 / 6 (sandal 5, net post 1) |
| 222542_60fps cam0 (…8F) | 5,769 | 1,826 (31.6 %) | 963 (53 %) | 7 / 5 (sandal 4, hand 1) |
| 222542_60fps cam1 (…8E) | 5,769 | 2,126 (36.9 %) | 1,499 (71 %) | 8 / 4 (sandal 3, hand 1) |

**Every one of the six new sessions has rallies in both cameras.**
Consecutive-frame candidate pairs that move 15–150 px number 777–1,291 per
camera. Runs of 6 or more such pairs number 22–92 per camera. No 30 s
window, in any camera, drops below a 0.079 candidate rate (0.0789, in
214831 cam1's final partial window, from the `summary.json` timeline). Over all 144
sampled candidates, 96 (66.7 %) are the real ball. The 48 false positives break down as follows:
  - 37 sandals or feet at the table edges.
  - 10 hands, wrists, or paddle edges.
  - 1 net post.

These are the same failure types found in 215350. So the next gain in
precision should come from a filter for sandals, skin, or track consistency,
not from the color band. Median disc depth is 1,451–2,000 mm. Depth
identifies the ball in 45–71 % of candidates, which agrees with 215350. The
median candidate radius is 6.4–9.2 px. The detector operates at about 240
frames per second, in one process, so a full pass over a session needs no
stride. Note: the file
`ball2d_orbbec_20260815_215350_cam0_CP0E8530008E.csv` was found overwritten
by a 5-row test run (09:24, 2026-08-27). The team made it again with the
documented command, and got the same result, 1,270/3,980
(`log_gated_215350_cam0_rerun.txt`).

With `--roi-margin 15`, the 215350 counts drop to 860 (21.6 %) / 631 (15.9 %)
(`work/perception/roi15/`). This comes at the cost of missing balls in the air past the
far edge. Ungated color-only detection gives near 100% "candidates" (pure
clutter). The old figure of "119/121 frames in a rally" has no clear source.
Ungated 214619 cam1 (a session with **no ball in flight**) gives exactly
119/121 (`work/perception/nomotion/log_214619_cam1.txt`). So that figure
described clutter, not a rally. Tracks are in
`work/perception/ball2d_<session>_<cam>.csv` (frame, host_time_s, u, v,
radius. Blank when nothing is found). A ball in flight moves 50–80 px per
frame at 20 fps and shows motion blur. The net rally length in the images is
limited by the roughly 20 fps rate.

**Depth (`sim/ball_3d.py`).** The depth PNGs are plain 16-bit greyscale
(`I;16`, no PNG text chunks, no EXIF data in the JPGs). `depth_png_unit` is
`millimeters_uint16`, `depth_scale` is 1.0 in every row, and about 99% of
values are non-zero. Median depth inside the ball disc, against an annulus
around it:
  - For 215350 cam0: the median disc depth is 1,649 mm. The disc reads more than 40 mm nearer than the annulus in 67% of hits.
  - For cam1: 1,754 mm, 50% of hits.

So the aligned depth **does identify the ball roughly half to
two-thirds of the time**. The rest of the time it reads the table behind the
ball (and false positives read the floor, at about 3.5 m). Output:
`work/perception/ball3d_<session>_<cam>.csv` (adds depth_mm, n_valid,
bg_depth_mm).

**What blocks metric XYZ / `ball_state`.** No intrinsics exist anywhere. These sources all lack fx, fy, cx, cy, the distortion values, and the cam0-to-cam1 extrinsics:
  - `session.json`.
  - `frames.jsonl`.
  - The tar file (which holds only images and the two JSON files).
  - The PNG/JPEG headers.
  - The HIM manifests (`calibration: []`, "No intrinsics/extrinsics available").

There are three ways to get them, and the team has done none of them yet:
  - (a) `pyorbbecsdk` (not on PyPI — build it from `github.com/orbbec/pyorbbecsdk`), with the same two cameras plugged in. Use `pipeline.get_camera_param()` on serials `CP0E8530008E` and `CP0E8530008F` at the 640×480 color profile (the aligned depth shares the color intrinsics).
  - (b) Checkerboard or ChArUco calibration of each camera at 640×480.
  - (c) If someone measures the ArUco marker edge length and the marker positions on the table, `solvePnP` per camera gives the table-frame pose. With 4 or more non-coplanar views, or `calibrateCamera` on the 8 markers over many frames, this also gives an estimate of the intrinsics. This is a weaker method, but it uses only the existing footage.

The team also still needs:
  - The marker size and table offsets.
  - The cam0-to-cam1 relation (through the shared table frame).
  - The clock offset between the Orbbec host clock and the IMU/Rokoko clocks.

Tooling: `~/him-tools/cv-venv` (python 3.12.8: numpy 2.5.2,
opencv-contrib-python 5.0.0.93, imageio 2.37.4, pillow 12.3.0). All outputs
are under `work/perception/` (git-ignored).

**Self-calibrated intrinsics (2026-08-27, `sim/estimate_intrinsics.py`,
results in `work/calib/`).** The team used the four `DICT_4X4_50` markers, plus the
aligned depth, plus the ITTF table size, per serial. From these, the team found the
camera height above the table (1.9 m, both serials) and the table plane to
about 1°. The team also found a focal length whose value depends on an unknown scale. The
team did **not** find the focal length on its own. A single view of one flat
rectangle cannot separate it from the unknown marker inset or size. The free
fit drifts (fy≈300, cy≈0). Changing the inset from 0 to 15 cm moves fx from
416 to 531 px. The change in the residual is less than 1 mm
(a verifier re-derived this). Use the center-fixed, square-pixel variant:
  - Camera …8E gives fx≈fy≈444 px.
  - Camera …8F gives fx≈482–501 px.
  - Both are about 67° HFOV, a possible value for an Orbbec RGB camera.

This has a ±15% scale caveat.
`sim/ball_3d.py --intrinsics work/calib/intrinsics_<serial>.json` then gives
X/Y/Z, in the camera frame and in the table frame. One tape measurement would settle this scale
question to within about 1%. This could be a measurement of a marker's printed edge
(the fit gives about 9.3 cm), or of its distance from the table corner. The camera model
(serial prefix CP0E85…) is still not confirmed.

