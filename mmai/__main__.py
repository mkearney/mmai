"""CLI: python -m mmai [daily|train|backtest|all|publish]"""
import sys

from .pipeline import run

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "daily"
    if mode not in ("daily", "train", "backtest", "all", "publish"):
        raise SystemExit("usage: python -m mmai [daily|train|backtest|all|publish]")
    run(mode)
