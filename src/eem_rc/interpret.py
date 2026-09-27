from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from .data import load_dataset
from .models import build_cnn, compile_cnn


def grad_ram(model: tf.keras.Model, samples: np.ndarray, layer_name: str = "conv2d_2") -> np.ndarray:
    probe = tf.keras.Model(model.inputs, [model.get_layer(layer_name).output, model.output])
    maps = []
    for sample in samples:
        batch = tf.convert_to_tensor(sample[None, ...], dtype=tf.float32)
        with tf.GradientTape() as tape:
            features, prediction = probe(batch, training=False)
        gradients = tape.gradient(prediction, features)
        weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
        activation = tf.nn.relu(tf.reduce_sum(weights * features, axis=-1, keepdims=True))
        activation = tf.image.resize(activation, (81, 81), method="bilinear")[0, ..., 0]
        maximum = tf.reduce_max(activation)
        activation = tf.where(maximum > 0, activation / maximum, activation)
        maps.append(np.flipud(activation.numpy()))
    return np.asarray(maps, dtype=np.float32)


def fit_full_model(
    data_dir: str | Path,
    output_dir: str | Path,
    epochs: int = 200,
    batch_size: int = 16,
    patience: int = 50,
) -> None:
    masked, _raw, y, files = load_dataset(data_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    tf.random.set_seed(0)
    np.random.seed(0)
    intensity_max = float(np.max(masked[..., 0]))
    x = masked.copy()
    x[..., 0] /= intensity_max
    scaler = MinMaxScaler().fit(y[:, None])
    y_scaled = scaler.transform(y[:, None])

    model = compile_cnn(build_cnn(channels=2))
    stopper = tf.keras.callbacks.EarlyStopping(
        monitor="loss", patience=patience, restore_best_weights=True
    )
    model.fit(x, y_scaled, epochs=epochs, batch_size=batch_size, callbacks=[stopper], verbose=2)
    model.save(output / "rcpm_full_data.keras")
    joblib.dump({"feature_max": intensity_max}, output / "feature_scaler.joblib")
    joblib.dump(scaler, output / "label_scaler.joblib")

    normalized_predictions = model.predict(x, batch_size=batch_size, verbose=0)
    predictions = scaler.inverse_transform(normalized_predictions).reshape(-1)
    np.save(output / "predictions_normalized.npy", normalized_predictions)
    pd.DataFrame({"filename": files, "measured": y, "predicted": predictions}).to_csv(
        output / "full_data_predictions.csv", index=False
    )

    feature_model = tf.keras.Model(model.inputs, model.get_layer("conv2d_2").output)
    feature_maps = feature_model.predict(x, batch_size=batch_size, verbose=0)
    flat = feature_maps.reshape(len(feature_maps), -1)
    components = min(100, len(flat) - 1, flat.shape[1] - 1)
    pca_features = PCA(n_components=components, svd_solver="arpack", random_state=0).fit_transform(flat)
    np.save(output / "cnn_features_pca.npy", pca_features)
    np.save(output / "grad_ram.npy", grad_ram(model, x))

