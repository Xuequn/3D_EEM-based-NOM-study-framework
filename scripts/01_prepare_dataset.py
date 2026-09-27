from __future__ import annotations

import argparse

from eem_rc.data import build_dataset, save_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the aligned mask-aware EEM dataset")
    parser.add_argument("--eem-dir", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--filename-column", default="filename")
    parser.add_argument("--label-column", default="label")
    args = parser.parse_args()
    dataset = build_dataset(args.eem_dir, args.labels, args.filename_column, args.label_column)
    save_dataset(args.output_dir, *dataset)
    masked, raw, y, files = dataset
    print(f"Saved {len(files)} samples: masked={masked.shape}, raw={raw.shape}, labels={y.shape}")


if __name__ == "__main__":
    main()

