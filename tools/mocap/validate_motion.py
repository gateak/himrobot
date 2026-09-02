"""Validate BVH motion clips before retargeting.

Usage: python tools/mocap/validate_motion.py --input <file.bvh|dir> --output <report.json>
        [--expected-fps 30] [--root Hips]

Per clip, checks parseable HIERARCHY/MOTION, root joint present, frame count > 0,
frame time within ±1% of expected fps, channel count consistent with hierarchy,
and no NaN/inf values. Exits 1 if any clip fails.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def parse_bvh(path: Path) -> dict:
    text = path.read_text()
    if "HIERARCHY" not in text or "MOTION" not in text:
        raise ValueError("missing HIERARCHY or MOTION section")
    hierarchy, motion = text.split("MOTION", 1)
    joints, channels = [], 0
    for line in hierarchy.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] in {"ROOT", "JOINT"}:
            joints.append(parts[1])
        elif parts[0] == "CHANNELS":
            channels += int(parts[1])
    lines = [l for l in motion.splitlines() if l.strip()]
    frames = int(lines[0].split(":")[1])
    frame_time = float(lines[1].split(":")[1])
    rows = lines[2:]
    bad = 0
    widths = set()
    for row in rows:
        values = [float(v) for v in row.split()]
        widths.add(len(values))
        bad += sum(1 for v in values if math.isnan(v) or math.isinf(v))
    return {"joints": joints, "channels": channels, "frames_declared": frames,
            "frames_present": len(rows), "frame_time": frame_time,
            "row_widths": sorted(widths), "non_finite_values": bad}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(path: Path, expected_fps: float, root: str) -> dict:
    report = {"file": path.name, "ok": True, "issues": [], "sha256": _sha256(path)}
    try:
        info = parse_bvh(path)
    except Exception as exc:  # report, don't crash the batch
        report["ok"] = False
        report["issues"].append(f"parse_error: {exc}")
        return report
    report.update(info)
    fps = 1.0 / info["frame_time"] if info["frame_time"] else 0.0
    report["fps"] = round(fps, 4)
    report["duration_s"] = round(info["frames_present"] * info["frame_time"], 3)
    if root not in info["joints"]:
        report["issues"].append(f"root_joint_missing: {root}")
    if info["frames_present"] == 0:
        report["issues"].append("no_frames")
    if info["frames_present"] != info["frames_declared"]:
        report["issues"].append("frame_count_mismatch")
    if abs(fps - expected_fps) > 0.01 * expected_fps:
        report["issues"].append(f"fps_unexpected: {fps:.3f} != {expected_fps}")
    if info["row_widths"] != [info["channels"]]:
        report["issues"].append(f"channel_width_mismatch: rows={info['row_widths']} hierarchy={info['channels']}")
    if info["non_finite_values"]:
        report["issues"].append(f"non_finite_values: {info['non_finite_values']}")
    report["ok"] = not report["issues"]
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-fps", type=float, default=30.0)
    parser.add_argument("--root", default="Hips")
    args = parser.parse_args()
    files = sorted(args.input.glob("*.bvh")) if args.input.is_dir() else [args.input]
    if not files:
        raise SystemExit(f"no .bvh files under {args.input}")
    clips = [check(f, args.expected_fps, args.root) for f in files]
    summary = {"clips": len(clips), "passed": sum(c["ok"] for c in clips),
               "total_frames": sum(c.get("frames_present", 0) for c in clips),
               "total_duration_s": round(sum(c.get("duration_s", 0) for c in clips), 3),
               "results": clips}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}))
    return 0 if summary["passed"] == summary["clips"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
