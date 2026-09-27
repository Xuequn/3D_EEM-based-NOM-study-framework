from __future__ import annotations

import argparse

from eem_rc.interpret import fit_full_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit the full-data RCPM and extract interpretation products")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=50)
    args = parser.parse_args()
    fit_full_model(args.data_dir, args.output_dir, args.epochs, args.batch_size, args.patience)


if __name__ == "__main__":
    main()

