"""Register this box's RL results as a private HIM dataset version so they survive machine termination.

Usage (on the box): python3 /workspace/himrobot/push_results.py
Collects work/tt/rl/ppo_*/{train_log.csv,best.pt,final.pt}, the BC warm start, the demo summary and the
logs, writes /workspace/push/manifest.json + summary.json, and runs `him dataset push --yes --json`.
"""
import glob
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = "/workspace/himrobot/work/tt/rl"
OUT = "/workspace/push"
HIM = os.path.expanduser("~/.local/bin/him")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def media_type(path):
    if path.endswith(".csv"):
        return "text/csv"
    if path.endswith(".json"):
        return "application/json"
    if path.endswith(".log"):
        return "text/plain"
    return "application/octet-stream"


def main():
    os.makedirs(OUT, exist_ok=True)
    files = []
    runs = sorted(glob.glob(ROOT + "/ppo_*"))
    for run in runs:
        for name in ("train_log.csv", "best.pt", "final.pt"):
            p = os.path.join(run, name)
            if os.path.isfile(p) and os.path.getsize(p) > 0:
                files.append(p)
    for extra in ("demos_seed0_summary.json", "bc_seed0.pt", "bc_seed0_mac.pt"):
        p = os.path.join(ROOT, extra)
        if os.path.isfile(p) and os.path.getsize(p) > 0:
            files.append(p)
    files += [p for p in sorted(glob.glob(ROOT + "/logs/*.log")) if os.path.getsize(p) > 0]

    summary = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "machine": "ssh_7c043322477649038603", "runs": {}}
    for run in runs:
        csv = os.path.join(run, "train_log.csv")
        last = None
        if os.path.isfile(csv):
            rows = [l.strip() for l in open(csv) if l.strip()]
            last = rows[-1] if len(rows) > 1 else None
        summary["runs"][os.path.basename(run)] = {
            "last_row": last,
            "has_best": os.path.isfile(os.path.join(run, "best.pt")),
            "has_final": os.path.isfile(os.path.join(run, "final.pt")),
        }
    sp = os.path.join(OUT, "summary.json")
    with open(sp, "w") as f:
        json.dump(summary, f, indent=1)
    files.append(sp)

    version = "0.1." + str(int(time.time()) // 60)
    uri = lambda p: "box://himrobot-rl/" + os.path.relpath(p, "/workspace")
    arts = [{"role": "metadata", "uri": uri(p), "sha256": sha256(p), "media_type": media_type(p)} for p in files]
    manifest = {
        "schema_version": "1.0.0",
        "dataset_slug": "himrobot-rl-checkpoints",
        "version": version,
        "display_name": "himrobot lane-1 PPO checkpoints and curves",
        "source_captures": ["tt-sim-rl-ssh_7c043322477649038603"],
        "streams": [{"key": "train_log", "modality": "other", "dtype": "float64", "shape": [-1, 6], "rate_hz": 1.0,
                     "time_basis": "host", "frame_id": None, "unit": None, "calibration_ref": None}],
        "episodes": {"count": len(runs), "index_artifact_uri": uri(sp)},
        "splits": {"all": [os.path.basename(r) for r in runs]},
        "artifacts": arts,
        "quality": {"note": "RL training outputs from tt_sim, not capture data"},
        "license": {"identifier": "Proprietary", "spdx_or_name": "Proprietary", "publication_authorized": False},
        "privacy": {"contains_audio": False, "contains_direct_personal_identifiers": False, "contains_images": False},
        "extensions": {"description": "PPO runs (cold s0-3, warm s10-13) on the tt_sim fixed-base G1 dev scene: "
                                      "train_log.csv per run, best/final checkpoints, BC warm start, logs."},
    }
    mp = os.path.join(OUT, "manifest.json")
    with open(mp, "w") as f:
        json.dump(manifest, f, indent=1)

    cmd = [HIM, "dataset", "push", mp, "--yes", "--json"]
    for a, p in zip(arts, files):
        cmd += ["--file", f"{a['uri']}={p}"]
    print(f"pushing {len(files)} files as {version}", file=sys.stderr)
    r = subprocess.run(cmd, capture_output=True, text=True)
    with open(os.path.join(OUT, f"push_{version}.log"), "w") as f:
        f.write(r.stdout + "\n--- stderr ---\n" + r.stderr)
    try:
        d = json.loads(r.stdout)
        print("PUSHED", d["status"], d.get("registry", {}).get("id"), version)
    except Exception:
        print("PUSH FAILED", r.stdout[-600:], r.stderr[-600:])
        sys.exit(1)


if __name__ == "__main__":
    main()
