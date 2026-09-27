from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage

from .data import load_dataset
from mda import discoverManifold, mda


def run_clustering(
    data_dir: str | Path,
    interpret_dir: str | Path,
    output_dir: str | Path,
    neighbors: int = 5,
    max_k: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _masked, _raw, measured, files = load_dataset(data_dir)
    interpretation = Path(interpret_dir)
    features = np.load(interpretation / "cnn_features_pca.npy")
    predicted = np.load(interpretation / "predictions_normalized.npy").reshape(-1, 1)
    if not (len(features) == len(predicted) == len(measured)):
        raise ValueError("Feature, prediction, and label arrays have inconsistent lengths")

    manifold_coordinate = discoverManifold(predicted, neighbors)
    embedding = np.asarray(mda(features, manifold_coordinate))
    if embedding.shape != (len(files), 2):
        raise ValueError(f"Expected a two-dimensional MDA embedding, got {embedding.shape}")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    embedding_frame = pd.DataFrame(
        {"filename": files, "MDA1": embedding[:, 0], "MDA2": embedding[:, 1], "label": measured}
    )
    embedding_frame.to_csv(output / "mda_embedding.csv", index=False)

    tree = linkage(embedding, method="ward", metric="euclidean")
    selection_rows = []
    assignments: dict[int, np.ndarray] = {}
    for k in range(2, max_k + 1):
        raw_clusters = fcluster(tree, t=k, criterion="maxclust")
        groups = [measured[raw_clusters == value] for value in np.unique(raw_clusters)]
        h_value, p_value = stats.kruskal(*groups)
        epsilon_squared = (h_value - k + 1) / (len(measured) - k)
        selection_rows.append(
            {"K": k, "H": h_value, "p_value": p_value, "epsilon_squared": epsilon_squared}
        )
        assignments[k] = raw_clusters

    selection = pd.DataFrame(selection_rows)
    selection.to_csv(output / "cluster_selection.csv", index=False)

    raw_k3 = assignments[3]
    ordered = sorted(np.unique(raw_k3), key=lambda value: measured[raw_k3 == value].mean())
    relabel = {old: new for new, old in enumerate(ordered, start=1)}
    c3 = np.asarray([relabel[value] for value in raw_k3])
    clusters = embedding_frame.copy()
    clusters["cluster"] = c3
    clusters.to_csv(output / "clusters_k3.csv", index=False)
    np.save(output / "ward_linkage.npy", tree)
    return selection, clusters
