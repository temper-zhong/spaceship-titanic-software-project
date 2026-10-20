"""Run the complete Spaceship Titanic workflow in a reproducible order."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def _run(command: list[str]) -> None:
    print(f"\n> {' '.join(command)}", flush=True)
    subprocess.run(command, check=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("raw data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use reduced tuning grids, three training folds, and smaller estimators.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    python = sys.executable

    common_paths = [
        "--data-dir",
        str(args.data_dir),
        "--output-dir",
        str(args.output_dir),
    ]
    quick = ["--quick"] if args.quick else []

    try:
        _run(
            [
                python,
                "-m",
                "src.eda",
                *common_paths,
                "--report-dir",
                str(args.report_dir),
            ]
        )
        _run(
            [
                python,
                "-m",
                "src.tune",
                *common_paths,
                "--folds",
                "3",
                "--seed",
                str(args.seed),
                *quick,
            ]
        )
        _run(
            [
                python,
                "-m",
                "src.train",
                *common_paths,
                "--folds",
                "3" if args.quick else "5",
                "--seed",
                str(args.seed),
                "--params-file",
                str(args.output_dir / "best_params.json"),
                *quick,
            ]
        )
        _run(
            [
                python,
                "-m",
                "src.analyze",
                *common_paths,
                "--report-dir",
                str(args.report_dir),
            ]
        )
    except subprocess.CalledProcessError as exc:
        print(f"ERROR: workflow stopped with exit code {exc.returncode}.", file=sys.stderr)
        return exc.returncode or 1

    print(f"\nComplete workflow finished in {time.perf_counter() - started:.1f}s.")
    print(f"Submission: {(args.output_dir / 'submission.csv').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
