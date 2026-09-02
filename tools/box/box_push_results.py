"""Register a box's training results as a private HIM dataset version so they survive machine termination.

Usage (on the box, from the repo root, with the him CLI logged in):
    /workspace/venv/bin/python tools/box/box_push_results.py --root work/tt/rl/param_v2 --machine ssh_xxx \\
        --description "hitter-v2 robustness sweep"

Collects <root>/<run>/{train_log.csv,best.pt,final.pt,RESULT.md}, <root>/*.md|*.json|*.log, <root>/logs/*.log
and <root>/scoreboard*/scoreboard.*, writes <out>/manifest.json + summary.json (last train_log row per run)
and runs `him dataset push --yes --json` against the dataset slug (default himrobot-rl-checkpoints). Safe
to re-run on a timer: every push is a new version (0.1.<minutes since epoch>).

--glob PATTERN (repeatable) adds extra files matched relative to --root, on top of the fixed set above --
e.g. an rsl_rl/Isaac Lab layout (tools/box/wbc_box_run.sh) that does not follow the <run>/train_log.csv
convention: --glob "wbc_v1/*/model_*.pt" --glob "wbc_v1/*/exported/*.npz". Omitting --glob leaves every
existing call (and its output) exactly as before.
"""
import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import time

RUN_FILES = ("train_log.csv", "best.pt", "final.pt", "RESULT.md")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def media_type(path: str) -> str:
    return {".csv": "text/csv", ".json": "application/json", ".log": "text/plain",
            ".md": "text/markdown"}.get(os.path.splitext(path)[1], "application/octet-stream")


def collect(root: str, extra_globs: tuple[str, ...] = ()) -> tuple[list[str], list[str]]:
    files = []
    runs = sorted(d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)
                  and os.path.isfile(os.path.join(d, "train_log.csv")))
    for run in runs:
        for name in RUN_FILES:
            p = os.path.join(run, name)
            if os.path.isfile(p) and os.path.getsize(p) > 0:
                files.append(p)
    for pattern in ("*.md", "*.json", "*.log", "logs/*.log", "scoreboard*/scoreboard.*") + tuple(extra_globs):
        files += [p for p in sorted(glob.glob(os.path.join(root, pattern)))
                  if os.path.isfile(p) and os.path.getsize(p) > 0]
    return runs, sorted(set(files))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="directory holding one sub-directory per training run")
    ap.add_argument("--machine", required=True, help="HIM machine id, recorded in the manifest")
    ap.add_argument("--out", default="/workspace/push", help="where manifest/summary/push logs are written")
    ap.add_argument("--slug", default="himrobot-rl-checkpoints")
    ap.add_argument("--description", default="tt_sim parameter-residual training runs")
    ap.add_argument("--him", default=os.path.expanduser("~/.local/bin/him"))
    ap.add_argument("--dry-run", action="store_true", help="write the manifest, do not push")
    ap.add_argument("--glob", action="append", default=[],
                    help="extra glob pattern relative to --root, on top of the fixed set; repeatable")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    runs, files = collect(args.root, tuple(args.glob))
    summary = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "machine": args.machine,
               "root": args.root, "runs": {}}
    for run in runs:
        rows = [l.strip() for l in open(os.path.join(run, "train_log.csv")) if l.strip()]
        summary["runs"][os.path.basename(run)] = {
            "header": rows[0] if rows else None,
            "last_row": rows[-1] if len(rows) > 1 else None,
            "has_best": os.path.isfile(os.path.join(run, "best.pt")),
            "has_final": os.path.isfile(os.path.join(run, "final.pt")),
        }
    sp = os.path.join(args.out, "summary.json")
    with open(sp, "w") as f:
        json.dump(summary, f, indent=1)
    files.append(sp)

    version = "0.1." + str(int(time.time()) // 60)
    root_abs = os.path.abspath(args.root)

    def uri(p: str) -> str:
        p = os.path.abspath(p)
        rel = os.path.relpath(p, root_abs) if p.startswith(root_abs + os.sep) else os.path.basename(p)
        return "box://himrobot-rl/" + rel.replace(os.sep, "/")

    arts = [{"role": "metadata", "uri": uri(p), "sha256": sha256(p), "media_type": media_type(p)} for p in files]
    manifest = {
        "schema_version": "1.0.0",
        "dataset_slug": args.slug,
        "version": version,
        "display_name": "himrobot tt_sim training checkpoints and curves",
        "source_captures": [f"tt-sim-rl-{args.machine}"],
        "streams": [{"key": "train_log", "modality": "other", "dtype": "float64", "shape": [-1, 9], "rate_hz": 1.0,
                     "time_basis": "host", "frame_id": None, "unit": None, "calibration_ref": None}],
        "episodes": {"count": len(runs), "index_artifact_uri": uri(sp)},
        "splits": {"all": [os.path.basename(r) for r in runs]},
        "artifacts": arts,
        "quality": {"note": "RL training outputs from tt_sim, not capture data"},
        "license": {"identifier": "Proprietary", "spdx_or_name": "Proprietary", "publication_authorized": False},
        "privacy": {"contains_audio": False, "contains_direct_personal_identifiers": False, "contains_images": False},
        "extensions": {"description": args.description},
    }
    mp = os.path.join(args.out, "manifest.json")
    with open(mp, "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"{len(files)} files from {len(runs)} runs -> {mp} (version {version})", file=sys.stderr)
    if args.dry_run:
        return

    cmd = [args.him, "dataset", "push", mp, "--yes", "--json"]
    for a, p in zip(arts, files):
        cmd += ["--file", f"{a['uri']}={p}"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    with open(os.path.join(args.out, f"push_{version}.log"), "w") as f:
        f.write(r.stdout + "\n--- stderr ---\n" + r.stderr)
    try:
        d = json.loads(r.stdout)
        print("PUSHED", d["status"], d.get("registry", {}).get("id"), version)
    except Exception:
        print("PUSH FAILED", r.stdout[-600:], r.stderr[-600:])
        sys.exit(1)


if __name__ == "__main__":
    main()
