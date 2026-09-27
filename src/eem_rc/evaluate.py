from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

from .data import load_dataset, source_from_filename
from .models import build_cnn, build_masked_conv_cnn, compile_cnn

SUPPORTED_MODELS = ("plsr", "rf", "xgboost", "zero_cnn", "mask_aware_cnn", "masked_conv_cnn")


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MSE": float(mse),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def _split_train_validation(indices: np.ndarray, fold: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(32 + fold)
    shuffled = rng.permutation(indices)
    validation_size = int(len(shuffled) * 0.2)
    return shuffled[validation_size:], shuffled[:validation_size]


def _cnn_input(masked: np.ndarray, model_name: str, scale: float) -> np.ndarray:
    if model_name == "zero_cnn":
        return (masked[..., :1] / scale).astype(np.float32)
    result = masked.copy().astype(np.float32)
    result[..., 0] /= scale
    return result


def _build_estimator(model_name: str):
    if model_name == "plsr":
        return PLSRegression(n_components=15)
    if model_name == "rf":
        return RandomForestRegressor(n_estimators=60, random_state=32)
    if model_name == "xgboost":
        return XGBRegressor(n_estimators=60, max_depth=3, learning_rate=0.005, random_state=32)
    raise ValueError(model_name)


def run_loso(
    data_dir: str | Path,
    output_dir: str | Path,
    model_names: list[str],
    epochs: int = 200,
    batch_size: int = 16,
    patience: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unknown = set(model_names) - set(SUPPORTED_MODELS)
    if unknown:
        raise ValueError(f"Unsupported model(s): {sorted(unknown)}")

    masked, _raw, y, files = load_dataset(data_dir)
    groups = np.asarray([source_from_filename(name) for name in files])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    predictions: list[pd.DataFrame] = []

    splitter = LeaveOneGroupOut()
    for fold, (development_idx, test_idx) in enumerate(splitter.split(masked, y, groups), start=1):
        train_idx, validation_idx = _split_train_validation(development_idx, fold - 1)
        test_source = str(np.unique(groups[test_idx])[0])
        intensity_max = float(np.max(masked[train_idx, ..., 0]))
        if intensity_max <= 0:
            raise ValueError(f"Non-positive training intensity maximum in fold {fold}")
        label_scaler = MinMaxScaler().fit(y[train_idx, None])

        for model_name in model_names:
            np.random.seed(0)
            random.seed(0)
            tf.random.set_seed(0)

            if model_name in {"plsr", "rf", "xgboost"}:
                x = (masked[..., 0] / intensity_max).reshape(len(masked), -1)
                estimator = _build_estimator(model_name)
                estimator.fit(x[train_idx], y[train_idx])
                subset_predictions = {
                    "train": estimator.predict(x[train_idx]).reshape(-1),
                    "validation": estimator.predict(x[validation_idx]).reshape(-1),
                    "test": estimator.predict(x[test_idx]).reshape(-1),
                }
            else:
                x = _cnn_input(masked, model_name, intensity_max)
                if model_name == "masked_conv_cnn":
                    model = compile_cnn(build_masked_conv_cnn())
                else:
                    model = compile_cnn(build_cnn(channels=x.shape[-1]))
                early_stop = tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=patience, restore_best_weights=True
                )
                model.fit(
                    x[train_idx],
                    label_scaler.transform(y[train_idx, None]),
                    validation_data=(x[validation_idx], label_scaler.transform(y[validation_idx, None])),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[early_stop],
                    verbose=2,
                )
                subset_predictions = {}
                for subset, idx in (("train", train_idx), ("validation", validation_idx), ("test", test_idx)):
                    normalized = model.predict(x[idx], batch_size=batch_size, verbose=0)
                    subset_predictions[subset] = label_scaler.inverse_transform(normalized).reshape(-1)
                model.save(output / f"{model_name}_fold{fold}_{test_source}.keras")
                tf.keras.backend.clear_session()

            for subset, idx in (("train", train_idx), ("validation", validation_idx), ("test", test_idx)):
                predicted = subset_predictions[subset]
                rows.append(
                    {
                        "model": model_name,
                        "fold": fold,
                        "held_out_source": test_source,
                        "subset": subset,
                        **metrics(y[idx], predicted),
                    }
                )
                predictions.append(
                    pd.DataFrame(
                        {
                            "model": model_name,
                            "fold": fold,
                            "held_out_source": test_source,
                            "subset": subset,
                            "filename": np.asarray(files)[idx],
                            "actual": y[idx],
                            "predicted": predicted,
                        }
                    )
                )

    metrics_frame = pd.DataFrame(rows)
    predictions_frame = pd.concat(predictions, ignore_index=True)
    metrics_frame.to_csv(output / "metrics_by_fold.csv", index=False)
    predictions_frame.to_csv(output / "predictions.csv", index=False)
    summary = metrics_frame.groupby(["model", "subset"])[["MSE", "MAE", "RMSE", "R2"]].agg(["mean", "std"])
    summary.to_csv(output / "metrics_summary.csv")
    (output / "run_config.json").write_text(
        json.dumps(
            {"models": model_names, "epochs": epochs, "batch_size": batch_size, "patience": patience},
            indent=2,
        ),
        encoding="utf-8",
    )
    return metrics_frame, predictions_frame

