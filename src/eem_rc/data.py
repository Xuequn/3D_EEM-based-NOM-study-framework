from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

EX_AXIS = np.arange(200, 605, 5, dtype=float)
EM_AXIS = np.arange(200, 605, 5, dtype=float)


def align_eem(path: str | Path) -> np.ndarray:
    """Align a workbook to the 81 x 81 grid by exact coordinate assignment."""
    frame = pd.read_excel(path, header=None)
    ex_raw = frame.iloc[1:, 0].to_numpy(dtype=float)
    em_raw = frame.iloc[0, 1:].to_numpy(dtype=float)
    values = frame.iloc[1:, 1:].to_numpy(dtype=float).T

    aligned = np.full((len(EM_AXIS), len(EX_AXIS)), np.nan, dtype=np.float32)
    ex_ok = np.isin(ex_raw, EX_AXIS)
    em_ok = np.isin(em_raw, EM_AXIS)
    ex_indices = np.searchsorted(EX_AXIS, ex_raw[ex_ok])
    em_indices = np.searchsorted(EM_AXIS, em_raw[em_ok])
    aligned[np.ix_(em_indices, ex_indices)] = values[np.ix_(em_ok, ex_ok)]
    return aligned


def build_dataset(
    eem_dir: str | Path,
    labels_path: str | Path,
    filename_col: str = "filename",
    label_col: str = "label",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    labels = pd.read_excel(labels_path)
    required = {filename_col, label_col}
    if not required.issubset(labels.columns):
        raise ValueError(f"Label workbook must contain columns {sorted(required)}")
    label_map = labels.set_index(filename_col)[label_col].to_dict()

    files = sorted(p for p in Path(eem_dir).iterdir() if p.suffix.lower() in {".xls", ".xlsx"})
    matched = [p for p in files if p.name in label_map]
    if not matched:
        raise ValueError("No EEM workbook matched the label table")

    raw = np.stack([align_eem(p) for p in matched]).astype(np.float32)
    y = np.asarray([label_map[p.name] for p in matched], dtype=np.float32)
    mask = np.isfinite(raw).astype(np.float32)
    zero = np.nan_to_num(raw, nan=0.0)
    masked = np.stack((zero, mask), axis=-1)
    return masked, raw, y, [p.name for p in matched]


def save_dataset(output_dir: str | Path, masked: np.ndarray, raw: np.ndarray, y: np.ndarray, files: list[str]) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "eem_features.npy", masked)
    np.save(output / "eem_features_raw.npy", raw)
    np.save(output / "eem_labels.npy", y)
    (output / "processed_files.txt").write_text("\n".join(files) + "\n", encoding="utf-8")


def load_dataset(data_dir: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    root = Path(data_dir)
    masked = np.load(root / "eem_features.npy")
    raw = np.load(root / "eem_features_raw.npy")
    y = np.load(root / "eem_labels.npy")
    files = [line for line in (root / "processed_files.txt").read_text(encoding="utf-8").splitlines() if line]
    if not (len(masked) == len(raw) == len(y) == len(files)):
        raise ValueError("Dataset arrays and filename list have inconsistent lengths")
    if masked.shape[1:] != (81, 81, 2):
        raise ValueError(f"Expected masked shape (n, 81, 81, 2), got {masked.shape}")
    return masked.astype(np.float32), raw.astype(np.float32), y.astype(np.float32), files


def source_from_filename(filename: str) -> str:
    for source in ("PPHA", "ESHA", "LHA"):
        if filename.upper().startswith(source):
            return source
    raise ValueError(f"Cannot infer HA source from filename: {filename}")

