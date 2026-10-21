"""Run controlled comparisons sequentially and keep artifacts outside code/."""

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

from reid.paths import DATA_DIR, RESULTS_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "experiments" / "night.json",
        help="JSON experiment settings; explicit CLI flags override them",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without training or reading data",
    )
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--prefix", default="reproduction")
    for flag, kind, default in [
        ("p", int, 4),
        ("k", int, 4),
        ("batch-size", int, 32),
        ("eps", float, 0.6),
        ("neighbors", int, 30),
        ("seed", int, 42),
        ("threads", int, 4),
    ]:
        parser.add_argument("--" + flag, type=kind, default=default)
    preliminary, _ = parser.parse_known_args()
    if preliminary.config:
        settings = json.loads(preliminary.config.read_text())
        allowed = {
            "epochs",
            "steps",
            "p",
            "k",
            "batch_size",
            "eps",
            "neighbors",
            "seed",
            "threads",
            "prefix",
        }
        if set(settings) - allowed:
            parser.error(
                "Unknown config settings: " + ", ".join(sorted(set(settings) - allowed))
            )
        parser.set_defaults(**settings)
    args = parser.parse_args()
    if min(args.epochs, args.steps, args.p, args.k, args.batch_size, args.threads) < 1:
        parser.error("Epochs, steps, P, K, batch size and threads must be positive")
    data, output = args.data.resolve(), args.output.resolve()
    variants = ("full", "baseline", "no_mlcor")
    source_dir = Path(__file__).resolve().parent

    def run(*arguments):
        command = [sys.executable, *map(str, arguments)]
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=source_dir, check=True)

    for variant in variants:
        folder = output / f"{args.prefix}_{variant}"
        run(
            "-m",
            "reid.run",
            "train",
            "--variant",
            variant,
            "--data",
            data,
            "--output",
            folder,
            "--epochs",
            args.epochs,
            "--steps",
            args.steps,
            "--p",
            args.p,
            "--k",
            args.k,
            "--batch-size",
            args.batch_size,
            "--eps",
            args.eps,
            "--neighbors",
            args.neighbors,
            "--seed",
            args.seed,
            "--threads",
            args.threads,
            "--device",
            args.device,
        )

    for variant in variants:
        folder = output / f"{args.prefix}_{variant}"
        run(
            "-m",
            "reid.run",
            "evaluate",
            "--data",
            data,
            "--checkpoint",
            folder / "best.pt",
            "--split",
            "testing",
            "--output",
            folder,
            "--batch-size",
            args.batch_size,
            "--threads",
            args.threads,
            "--device",
            args.device,
        )
    run(
        "-m",
        "reid.report",
        "--data",
        data,
        "--root",
        output,
        "--variants",
        *(f"{args.prefix}_{variant}" for variant in variants),
    )


if __name__ == "__main__":
    main()
