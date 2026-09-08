"""Single entry point for protocol-v1 training and held-out testing."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs/protocol_v1.json"))
    parser.add_argument("--model", choices=["bipulseformer", "physformer"], default="bipulseformer")
    parser.add_argument("--mode", choices=["cross", "intra"], default="cross")
    parser.add_argument("--source", choices=["PURE", "UBFC-rPPG"], required=True)
    parser.add_argument("--target", choices=["PURE", "UBFC-rPPG"])
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--target-root")
    parser.add_argument("--output", required=True, help="New, empty experiment directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true", help="Validate protocol and split without loading video or training")
    args = parser.parse_args()
    if args.mode == "cross" and (not args.target or not args.target_root or args.target == args.source):
        parser.error("cross requires a different --target and --target-root")
    if args.mode == "intra" and (args.target or args.target_root):
        parser.error("intra uses the source held-out subjects; omit target arguments")
    if args.workers < 0:
        parser.error("workers must be nonnegative")
    if args.threads < 1:
        parser.error('threads must be positive')
    from src.experiment import run
    run(args)


if __name__ == "__main__":
    main()
