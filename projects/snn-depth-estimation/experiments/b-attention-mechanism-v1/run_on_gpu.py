"""One registered B cell per allocation; isolated preflight and production."""
import argparse
import os
import sys

from settings import EXP, LEGACY, SEEDS, VARIANTS, load, run_folder, add_input_arguments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=SEEDS, required=True)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--resume', action='store_true')
    add_input_arguments(parser)
    args = parser.parse_args()
    # Stage root is configured by deployment. Runs never share checkpoints.
    os.environ['STEREO_RUN_ROOT'] = str(run_folder(
        os.environ['STEREO_RUN_ROOT'], args.seed, args.variant))
    sys.argv = [sys.argv[0], '--arm', 'B', *(['--resume'] if args.resume else [])]
    runner = load('b_matched_runner', LEGACY / 'run_on_gpu.py')
    worker_args = ('--seed', str(args.seed), '--variant', args.variant)
    if args.input_mode != 'legacy':
        worker_args += ('--input-mode', args.input_mode, '--preload-gib', str(args.preload_gib),
                        '--input-workers', str(args.input_workers), '--prefetch-depth', str(args.prefetch_depth))
    runner.main(worker_path=EXP / 'worker.py', worker_args=worker_args)


if __name__ == '__main__':
    main()
