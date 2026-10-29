#!/usr/bin/env python3
# CoinFT sensor visualization, migrated from coinft/teensy_mlp_workspace/visualization.
"""Compare left and right CoinFT values on six shared-axis waveforms."""

from _coinft_live_plot import build_parser, run_axis_compare


if __name__ == "__main__":
    args = build_parser(__doc__).parse_args()
    if args.history <= 0:
        raise SystemExit("--history must be positive")
    raise SystemExit(run_axis_compare(args))
