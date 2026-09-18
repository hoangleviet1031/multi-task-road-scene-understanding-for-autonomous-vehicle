"""Command-line interface for Milestone 2 baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from roadsense.training.config import load_experiment_config
from roadsense.training.engine import run_evaluation, run_training
from roadsense.training.smoke import run_smoke_test


def _train(args: argparse.Namespace) -> int:
    """Execute the train subcommand.

    Args:
        args: Parsed config and optional resume checkpoint.

    Returns:
        Zero on successful training.
    """

    summary = run_training(load_experiment_config(args.config), args.resume)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    """Execute checkpoint evaluation.

    Args:
        args: Parsed experiment config and checkpoint path.

    Returns:
        Zero after metrics are written.
    """

    report = run_evaluation(load_experiment_config(args.config), args.checkpoint)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def _smoke(args: argparse.Namespace) -> int:
    """Execute framework-only forward/backward checks.

    Args:
        args: Parsed output directory.

    Returns:
        Zero when all three baselines produce finite losses.
    """

    report = run_smoke_test(args.output)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the Milestone 2 command parser.

    Returns:
        Parser with train, evaluate, and smoke-test subcommands.
    """

    parser = argparse.ArgumentParser(
        prog="roadsense-train",
        description="Train and evaluate independent RoadSense baselines.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train one configured baseline.")
    train.add_argument("--config", required=True)
    train.add_argument("--resume")
    train.set_defaults(handler=_train)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate one checkpoint.")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.set_defaults(handler=_evaluate)

    smoke = subparsers.add_parser(
        "smoke-test", help="Run CPU forward/backward checks for all baselines."
    )
    smoke.add_argument("--output", required=True)
    smoke.set_defaults(handler=_smoke)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse command arguments and return a process exit status.

    Args:
        argv: Optional explicit argument sequence.

    Returns:
        Zero for success, two for a handled configuration/runtime error.
    """

    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
