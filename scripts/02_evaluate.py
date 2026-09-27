from __future__ import annotations

import argparse

from eem_rc.evaluate import SUPPORTED_MODELS, run_loso


def main() -> None:
    parser = argparse.ArgumentParser(description="Run leave-one-source-out model evaluation")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--models", nargs="+", choices=SUPPORTED_MODELS, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=50)
    args = parser.parse_args()
    metrics, _ = run_loso(
        args.data_dir, args.output_dir, args.models, args.epochs, args.batch_size, args.patience
    )
    print(metrics[metrics["subset"] == "test"].to_string(index=False))


if __name__ == "__main__":
    main()

