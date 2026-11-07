"""Opt-in B controls on the unchanged matched 30k training loop."""
import argparse
from functools import partial
import sys

from settings import EXP, LEGACY, SEEDS, VARIANTS, load, add_input_arguments


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--seed', type=int, choices=SEEDS, required=True)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    add_input_arguments(parser)
    args, remaining = parser.parse_known_args()
    sys.path.insert(0, str(EXP.parents[1]))
    from core.attention_encoder_controls import install, VARIANTS as RULES
    contract = load('b_seed_contract', EXP.parent / 'spiking-attention-placement-phase2/protocol.py')
    worker = load('b_matched_worker', LEGACY / 'worker.py')
    # The inherited parser still owns paths, --preflight and --stop-at.
    sys.argv = [sys.argv[0], *remaining]
    forward, backward = RULES[args.variant]
    pipeline_options = {}
    if args.input_mode != 'legacy':
        from input_pipeline import InputPipeline
        pipeline_options['input_pipeline'] = InputPipeline(
            args.input_mode, args.preload_gib, args.input_workers, args.prefetch_depth, args.seed)
    return worker.main(
        **pipeline_options,
        seed=args.seed,
        model_install=partial(install, variant=args.variant, seed=args.seed),
        prefetch_factory=partial(contract.Prefetch, seed=args.seed),
        config_extension=dict(study='b-attention-mechanism-v1',
                              coding=args.variant, forward_rule=forward,
                              backward_rule=backward, placement_seed=args.seed + 783,
                              augmentation_seed=args.seed, derivative_order=1))


if __name__ == '__main__':
    raise SystemExit(main())
