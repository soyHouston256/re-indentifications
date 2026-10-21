"""Run controlled comparisons sequentially and keep artifacts outside code/."""
import argparse
import subprocess
import sys
from pathlib import Path

from reid.paths import DATA_DIR, RESULTS_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DATA_DIR)
    parser.add_argument('--output', type=Path, default=RESULTS_DIR)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--prefix', default='pilot')
    parser.add_argument('--skip-full', action='store_true',
                        help='Reuse a completed full-model run with the same budget')
    args = parser.parse_args()
    data, output = args.data.resolve(), args.output.resolve()
    variants = ('full', 'baseline', 'no_mlcor')
    source_dir = Path(__file__).resolve().parent

    def run(*arguments):
        subprocess.run([sys.executable, *map(str, arguments)], cwd=source_dir, check=True)

    for variant in variants:
        folder = output / f'{args.prefix}_{variant}'
        if variant == 'full' and args.skip_full:
            if not (folder / 'best.pt').is_file():
                parser.error(f'Missing checkpoint: {folder / "best.pt"}')
            continue
        run('-m', 'reid.run', 'train', '--variant', variant,
            '--data', data, '--output', folder,
            '--epochs', args.epochs, '--steps', args.steps,
            '--p', 4, '--k', 4, '--batch-size', 32, '--device', args.device)

    # All model selection finishes before any testing split is opened.
    for variant in variants:
        folder = output / f'{args.prefix}_{variant}'
        run('-m', 'reid.run', 'evaluate', '--data', data,
            '--checkpoint', folder / 'best.pt', '--split', 'testing',
            '--output', folder, '--device', args.device)
    run('-m', 'reid.report', '--data', data, '--root', output,
        '--variants', *(f'{args.prefix}_{variant}' for variant in variants))


if __name__ == '__main__':
    main()
