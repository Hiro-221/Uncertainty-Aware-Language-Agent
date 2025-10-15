import argparse
import json
import sys
from typing import Tuple, Optional


def safe_float(value) -> Optional[float]:
    if isinstance(value, bool):
        # Treat booleans as 1.0/0.0 for reward
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except Exception:
        return None


def aggregate_file(path: str) -> Tuple[int, float, int, float]:
    """
    Return tuple: (f1_count, f1_sum, reward_count, reward_sum)
    """
    f1_count = 0
    f1_sum = 0.0
    reward_count = 0
    reward_sum = 0.0

    open_stream = sys.stdin if path == "-" else open(path, "r", encoding="utf-8")
    try:
        for line in open_stream:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            if "f1" in obj:
                f1_val = safe_float(obj["f1"])
                if f1_val is not None:
                    f1_sum += f1_val
                    f1_count += 1

            if "reward" in obj:
                reward_val = safe_float(obj["reward"])
                if reward_val is not None:
                    reward_sum += reward_val
                    reward_count += 1
    finally:
        if open_stream is not sys.stdin:
            open_stream.close()

    return f1_count, f1_sum, reward_count, reward_sum


def format_avg(sum_val: float, count: int) -> str:
    if count == 0:
        return "n/a"
    return f"{(sum_val / count):.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute average f1 and reward from JSONL files.")
    parser.add_argument("files", nargs="+", help="One or more .jsonl files, or '-' for STDIN")
    args = parser.parse_args()

    total_f1_count = 0
    total_f1_sum = 0.0
    total_reward_count = 0
    total_reward_sum = 0.0

    for path in args.files:
        f1_c, f1_s, r_c, r_s = aggregate_file(path)
        total_f1_count += f1_c
        total_f1_sum += f1_s
        total_reward_count += r_c
        total_reward_sum += r_s

        print(f"File: {path}")
        print(f"  f1_avg: {format_avg(f1_s, f1_c)} (n={f1_c})")
        print(f"  reward_avg: {format_avg(r_s, r_c)} (n={r_c})")

    if len(args.files) > 1:
        print("Overall:")
        print(f"  f1_avg: {format_avg(total_f1_sum, total_f1_count)} (n={total_f1_count})")
        print(f"  reward_avg: {format_avg(total_reward_sum, total_reward_count)} (n={total_reward_count})")


if __name__ == "__main__":
    main()