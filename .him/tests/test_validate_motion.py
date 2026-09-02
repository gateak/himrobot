import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "mocap" / "validate_motion.py"

HEADER = """HIERARCHY
ROOT Hips
{
\tOFFSET 0 90 0
\tCHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation Zrotation
\tJOINT Spine
\t{
\t\tOFFSET 0 10 0
\t\tCHANNELS 3 Xrotation Yrotation Zrotation
\t\tEnd Site
\t\t{
\t\t\tOFFSET 0 10 0
\t\t}
\t}
}
MOTION
Frames: 3
Frame Time: 0.033333
"""


def run(tmp_path: Path, rows: str) -> tuple[int, dict]:
    clip = tmp_path / "clip.bvh"
    clip.write_text(HEADER + rows)
    report = tmp_path / "report.json"
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--input", str(clip), "--output", str(report)],
        capture_output=True, text=True, check=False,
    )
    return completed.returncode, json.loads(report.read_text())


def test_good_clip_passes(tmp_path: Path) -> None:
    code, report = run(tmp_path, "0 90 0 0 0 0 0 0 0\n0 91 0 1 0 0 0 0 0\n0 92 0 2 0 0 0 0 0\n")
    assert code == 0
    assert report["passed"] == 1 and report["total_frames"] == 3
    assert abs(report["results"][0]["fps"] - 30.0) < 0.01


def test_non_finite_values_fail(tmp_path: Path) -> None:
    code, report = run(tmp_path, "0 90 0 0 0 0 0 0 0\n0 nan 0 1 0 0 0 0 0\n0 92 0 2 0 0 0 0 0\n")
    assert code == 1
    assert any("non_finite_values" in issue for issue in report["results"][0]["issues"])
