from __future__ import annotations

import argparse
import json

from src.pipeline import run_all


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Network anomaly detection pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_all_parser = subparsers.add_parser("run-all", help="Execute all pipeline phases")
    run_all_parser.add_argument("--data-path", type=str, default=None, help="Path to raw dataset CSV/parquet")
    run_all_parser.add_argument("--synthetic-rows", type=int, default=5000, help="Rows for synthetic dataset fallback")
    run_all_parser.add_argument("--max-rows", type=int, default=None, help="Optional row cap for large real datasets")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run-all":
        artifacts = run_all(data_path=args.data_path, synthetic_rows=args.synthetic_rows, max_rows=args.max_rows)
        print(json.dumps(artifacts.metrics, indent=2))


if __name__ == "__main__":
    main()
