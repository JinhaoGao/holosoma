"""Command-line preflight for OMOMO motion data and object assets."""

from __future__ import annotations

import argparse
from pathlib import Path

from holosoma_retargeting.data_utils.omomo import preflight_omomo_dataset


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, help="Directory containing OMOMO .pt files")
    parser.add_argument("--asset-root", type=Path, help="Directory containing one subdirectory per object")
    parser.add_argument("--height-file", type=Path, help="Override the OMOMO height_dict.pkl path")
    parser.add_argument(
        "--skip-tensor-validation",
        action="store_true",
        help="Only inspect filenames, heights, and assets",
    )
    parser.add_argument("--output", type=Path, help="Write the JSON report to this path")
    return parser.parse_args()


def main() -> None:
    """Run preflight and exit non-zero when any issue is found."""

    args = _parse_args()
    report = preflight_omomo_dataset(
        args.data_dir,
        asset_root=args.asset_root,
        height_file=args.height_file,
        validate_tensors=not args.skip_tensor_validation,
    )
    payload = report.to_json()
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{payload}\n", encoding="utf-8")
    if not report.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
