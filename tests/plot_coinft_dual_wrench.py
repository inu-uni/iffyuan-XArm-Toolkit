#!/usr/bin/env python3
# CoinFT sensor visualization, migrated from coinft/teensy_mlp_workspace/visualization.
"""Plot left and right CoinFT force and moment waveforms in separate panels."""

from _coinft_live_plot import build_parser, run_dual


if __name__ == "__main__":
    args = build_parser(__doc__).parse_args()
    if args.history <= 0:
        raise SystemExit("--history must be positive")
    raise SystemExit(run_dual(args))
