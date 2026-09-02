#!/usr/bin/env python3
"""MJCF -> USD for the WBC asset (docs/design/whole_body_v1.md sec 2,
"Robot asset"), box-side only. The same six lines Isaac Lab's own
``scripts/tools/convert_mjcf.py`` runs -- but that file ships with the
IsaacLab *repository checkout*, not with the pip-installed ``isaaclab``
package, and the ``him-robotics`` image's checkout location is unknown, so
the conversion is driven here through the installed API
(``isaaclab.sim.converters.MjcfConverter``) instead of by path.

    /workspace/venv/bin/python -m himrobot.wbc.isaac.scripts.convert_mjcf \\
        assets/wbc/g1_29dof_paddle_free.xml /workspace/assets/g1_29dof_paddle.usd --headless

Never run in this repo's sandbox (no Isaac Sim); the converter cfg fields
used below (asset_path/usd_dir/usd_file_name/force_usd_conversion/
make_instanceable on the base cfg, fix_base/import_sites on the MJCF one)
are the v2.1.0 ones -- unverified here, same caveat as env_cfg.py.
"""
from __future__ import annotations

import argparse
import os


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="MJCF path (assets/wbc/g1_29dof_paddle_free.xml)")
    ap.add_argument("output", help="USD path to write")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--fix-base", action="store_true", help="weld the base (NOT for the WBC: the pelvis is free)")
    args = ap.parse_args(argv)

    mjcf_path = os.path.abspath(args.input)
    if not os.path.isfile(mjcf_path):
        raise SystemExit(f"MJCF not found: {mjcf_path}")
    dest_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=args.headless)
    simulation_app = app_launcher.app

    from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg

    cfg = MjcfConverterCfg(
        asset_path=mjcf_path,
        usd_dir=os.path.dirname(dest_path),
        usd_file_name=os.path.basename(dest_path),
        fix_base=args.fix_base,
        import_sites=True,
        force_usd_conversion=True,
        make_instanceable=False,
    )
    converter = MjcfConverter(cfg)
    print(f"MJCF: {mjcf_path}\nUSD:  {converter.usd_path}")
    simulation_app.close()
    if not os.path.isfile(converter.usd_path):
        raise SystemExit(f"converter reported {converter.usd_path} but the file is missing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
