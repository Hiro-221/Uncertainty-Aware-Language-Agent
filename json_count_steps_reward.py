#!/usr/bin/env python3
"""
Count entries in JSONL files where `steps != 2` and `reward == false`.

Usage:
  python json_count_steps_reward.py /path/to/file.jsonl [more.jsonl ...]

Notes:
  - Lines that are empty or not valid JSON objects are ignored.
  - Only counts lines where both conditions hold: steps != 2 AND reward is False.
  - If multiple files are provided, prints per-file counts and a total.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Tuple


def count_in_file(file_path: Path) -> Tuple[int, int]:
    """Return a tuple: (matching_count, total_parsed_lines) for a JSONL file.

    total_parsed_lines counts only lines that parsed as JSON objects.
    """
    matching = 0
    parsed = 0

    try:
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    # Skip lines that are not valid JSON
                    continue

                if not isinstance(obj, dict):
                    # Only consider JSON objects
                    continue

                parsed += 1

                steps_value = obj.get("steps")
                reward_value = obj.get("reward")

                if steps_value != 2 and reward_value is False:
                    matching += 1
    except FileNotFoundError:
        print(f"Error: file not found: {file_path}", file=sys.stderr)
    except PermissionError:
        print(f"Error: permission denied: {file_path}", file=sys.stderr)

    return matching, parsed


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count JSONL entries where steps != 2 and reward == false",
    )
    parser.add_argument(
        "files",
        metavar="JSONL",
        nargs="+",
        help="Path(s) to JSONL file(s) to process",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print the total (suppress per-file counts)",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    total_matching = 0
    total_parsed = 0

    for file_str in args.files:
        path = Path(file_str)
        matching, parsed = count_in_file(path)
        total_matching += matching
        total_parsed += parsed
        if not args.quiet and path.exists():
            print(f"{path}: {matching} (out of {parsed} valid JSON lines)")

    if len(args.files) > 1 or args.quiet:
        print(f"TOTAL: {total_matching}")
    else:
        # Single file: printing only the count is concise
        print(total_matching)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))










