from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from .data import EX_AXIS, EM_AXIS, load_dataset


def generalized_2dcos(spectra: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculate synchronous and asynchronous generalized 2D-COS maps."""
    centered = spectra - spectra.mean(axis=0, keepdims=True)
    sample_count = len(centered)
    if sample_count < 3:
        raise ValueError("At least three spectra are required for 2D-COS")
    noda = np.zeros((sample_count, sample_count), dtype=float)
    row, column = np.indices(noda.shape)
    off_diagonal = row != column
    noda[off_diagonal] = 1.0 / (np.pi * (column[off_diagonal] - row[off_diagonal]))
    synchronous = centered.T @ centered / (sample_count - 1)
    asynchronous = centered.T @ noda @ centered / (sample_count - 1)
    return synchronous, asynchronous


def _uniform_spectra(spectra: np.ndarray, perturbation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(perturbation)
    perturbation = perturbation[order]
    spectra = spectra[order]
    unique, inverse = np.unique(perturbation, return_inverse=True)
    if len(unique) < len(perturbation):
        averaged = np.stack([spectra[inverse == index].mean(axis=0) for index in range(len(unique))])
        spectra, perturbation = averaged, unique
    uniform = np.linspace(perturbation.min(), perturbation.max(), len(perturbation))
    interpolated = interp1d(perturbation, spectra, axis=0, kind="linear")(uniform)
    return interpolated, uniform


def run_twodcos(
    data_dir: str | Path,
    clusters_path: str | Path,
    output_dir: str | Path,
    emissions: tuple[float, ...] = (425.0, 490.0),
) -> None:
    _masked, raw, labels, files = load_dataset(data_dir)
    cluster_frame = pd.read_csv(clusters_path)
    required = {"filename", "cluster"}
    if not required.issubset(cluster_frame.columns):
        raise ValueError(f"Cluster file must contain {sorted(required)}")

    file_to_index = {name: index for index, name in enumerate(files)}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for first, second in ((1, 2), (2, 3)):
        pair = cluster_frame[cluster_frame["cluster"].isin((first, second))]
        indices = np.asarray([file_to_index[name] for name in pair["filename"]], dtype=int)
        for target_emission in emissions:
            emission_index = int(np.argmin(np.abs(EM_AXIS - target_emission)))
            # Stored EEM orientation is (emission, excitation).
            excitation_spectra = raw[indices, emission_index, :]
            excitation_spectra = np.nan_to_num(excitation_spectra, nan=0.0)
            interpolated, uniform = _uniform_spectra(excitation_spectra, labels[indices])
            synchronous, asynchronous = generalized_2dcos(interpolated)
            stem = f"C{first}_C{second}_Em{int(EM_AXIS[emission_index])}"
            np.savez_compressed(
                output / f"{stem}.npz",
                excitation_axis=EX_AXIS,
                reducing_capacity=uniform,
                synchronous=synchronous,
                asynchronous=asynchronous,
            )

