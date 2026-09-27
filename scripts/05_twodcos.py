from __future__ import annotations

import argparse

from eem_rc.twodcos import run_twodcos


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cluster-pair generalized 2D-COS")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--clusters", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--emissions", nargs="+", type=float, default=[425.0, 490.0])
    args = parser.parse_args()
    run_twodcos(args.data_dir, args.clusters, args.output_dir, tuple(args.emissions))


if __name__ == "__main__":
    main()

