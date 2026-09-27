from __future__ import annotations

import argparse

from eem_rc.clustering import run_clustering


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MDA and Ward hierarchical clustering")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--interpret-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--max-k", type=int, default=8)
    args = parser.parse_args()
    selection, clusters = run_clustering(
        args.data_dir, args.interpret_dir, args.output_dir, args.neighbors, args.max_k
    )
    print(selection.to_string(index=False))
    print("\nK=3 cluster means:")
    print(clusters.groupby("cluster")["label"].mean().to_string())


if __name__ == "__main__":
    main()

