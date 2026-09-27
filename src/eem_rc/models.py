from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Model, layers


def build_cnn(channels: int = 1) -> Model:
    inputs = tf.keras.Input((81, 81, channels), name="eem")
    x = layers.Conv2D(32, 3, padding="same", activation="relu", name="conv2d_0")(inputs)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu", name="conv2d_1")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu", name="conv2d_2")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(1)(x)
    return Model(inputs, outputs)


def build_masked_conv_cnn() -> Model:
    inputs = tf.keras.Input((81, 81, 2), name="eem_and_mask")
    data = inputs[..., :1]
    mask = inputs[..., 1:]

    x = layers.Conv2D(32, 3, padding="same", activation="relu", name="conv2d_0")(data)
    x = layers.Multiply()([x, mask])
    x = layers.MaxPooling2D(2)(x)
    mask = layers.MaxPooling2D(2)(mask)

    x = layers.Conv2D(64, 3, padding="same", activation="relu", name="conv2d_1")(x)
    x = layers.Multiply()([x, mask])
    x = layers.MaxPooling2D(2)(x)
    mask = layers.MaxPooling2D(2)(mask)

    x = layers.Conv2D(64, 3, padding="same", activation="relu", name="conv2d_2")(x)
    x = layers.Multiply()([x, mask])
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    return Model(inputs, layers.Dense(1)(x))


def compile_cnn(model: Model) -> Model:
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mean_squared_error", metrics=["mae"])
    return model

