"""Command-line entry points for Milestone 1 workflows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from roadsense.data.audit import audit_dataset
from roadsense.data.bdd100k import BDD100KDataset
from roadsense.data.config import load_dataset_config
from roadsense.data.metadata import metadata_records_from_dataset
from roadsense.data.split import create_group_stratified_split
from roadsense.self_check import run_self_check
from roadsense.visualization import save_sample_visualization


def _write_json(value: Any, path: str | Path) -> Path:
    """Write a serializable object as indented UTF-8 JSON.

    Args:
        value: JSON-serializable object.
        path: Destination file.

    Returns:
        Absolute path of the written report.
    """

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
    return destination


def _dataset(args: argparse.Namespace) -> BDD100KDataset:
    """Construct a joint dataset from common CLI arguments.

    Args:
        args: Namespace containing ``config`` and ``split``.

    Returns:
        Initialized BDD100K dataset.
    """

    return BDD100KDataset(load_dataset_config(args.config), args.split)


def _run_audit(args: argparse.Namespace) -> int:
    """Execute the audit subcommand and return a process status code.

    Args:
        args: Parsed audit arguments.

    Returns:
        Zero when no hard errors are found, otherwise two.
    """

    report = audit_dataset(_dataset(args), limit=args.limit)
    destination = _write_json(report, args.output)
    print(json.dumps({"report": str(destination), "ok": report["ok"]}, indent=2))
    return 0 if report["ok"] else 2


def _run_build_split(args: argparse.Namespace) -> int:
    """Execute leakage-safe split generation.

    Args:
        args: Parsed split arguments.

    Returns:
        Zero after a manifest is written successfully.
    """

    dataset = _dataset(args)
    manifest = create_group_stratified_split(
        metadata_records_from_dataset(dataset),
        dev_ratio=args.dev_ratio,
        seed=args.seed,
    )
    manifest["source_split"] = args.split
    manifest["config"] = str(Path(args.config).expanduser().resolve())
    destination = _write_json(manifest, args.output)
    print(
        json.dumps(
            {
                "manifest": str(destination),
                "train_count": manifest["train_count"],
                "dev_count": manifest["dev_count"],
                "actual_dev_ratio": manifest["actual_dev_ratio"],
            },
            indent=2,
        )
    )
    return 0


def _run_visualize(args: argparse.Namespace) -> int:
    """Render one indexed joint sample.

    Args:
        args: Parsed visualization arguments.

    Returns:
        Zero after the image is saved.
    """

    dataset = _dataset(args)
    if not -len(dataset) <= args.index < len(dataset):
        raise IndexError(f"Index {args.index} outside dataset of size {len(dataset)}.")
    sample = dataset[args.index]
    destination = save_sample_visualization(sample, args.output)
    print(json.dumps({"sample": sample.name, "image": str(destination)}, indent=2))
    return 0


def _run_self_check(args: argparse.Namespace) -> int:
    """Execute synthetic end-to-end verification.

    Args:
        args: Parsed self-check arguments.

    Returns:
        Zero when all checks pass.
    """

    report = run_self_check(args.output)
    print(json.dumps({"ok": report["ok"], **report["artifacts"]}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Create the RoadSense command-line parser.

    Returns:
        Configured parser with audit, split, visualization, and self-check
        subcommands.
    """

    parser = argparse.ArgumentParser(
        prog="roadsense",
        description="RoadSense Milestone 1 data and metric utilities.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="Audit a configured BDD100K split.")
    audit.add_argument("--config", required=True)
    audit.add_argument("--split", default="train")
    audit.add_argument("--limit", type=int, default=500)
    audit.add_argument("--output", required=True)
    audit.set_defaults(handler=_run_audit)

    split = subparsers.add_parser(
        "build-split", help="Create a group-aware train/development manifest."
    )
    split.add_argument("--config", required=True)
    split.add_argument("--split", default="train")
    split.add_argument("--dev-ratio", type=float, default=0.1)
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--output", required=True)
    split.set_defaults(handler=_run_build_split)

    visualize = subparsers.add_parser(
        "visualize", help="Render boxes and masks for one joint sample."
    )
    visualize.add_argument("--config", required=True)
    visualize.add_argument("--split", default="train")
    visualize.add_argument("--index", type=int, default=0)
    visualize.add_argument("--output", required=True)
    visualize.set_defaults(handler=_run_visualize)

    self_check = subparsers.add_parser(
        "self-check", help="Verify Milestone 1 without downloading BDD100K."
    )
    self_check.add_argument("--output", required=True)
    self_check.set_defaults(handler=_run_self_check)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments and execute a Milestone 1 command.

    Args:
        argv: Optional argument sequence. ``None`` reads from the process CLI.

    Returns:
        Process exit status; zero means success.
    """

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (FileNotFoundError, ValueError, IndexError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
