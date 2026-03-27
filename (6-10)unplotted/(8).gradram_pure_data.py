"""
Generate Grad-RAM heatmaps for EEM data visualization.
Visualizes which regions of the EEM data are important for model predictions.
"""
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
from matplotlib import cm
from PIL import Image
import matplotlib.pyplot as plt
import os

def prepare_input(array):
    """Prepare 81x81 matrix input"""
    # Ensure input shape is (1, 81, 81, 1)
    if array.ndim == 2:
        array = np.expand_dims(array, axis=-1)   # Add channel dimension
    array = np.expand_dims(array, axis=0)  # Add batch dimension
    return array.astype('float32')


def gradram_heatmap(input_array, model, last_conv_layer_name):
    """Generate Grad-CAM heatmap for 81x81 matrix model"""
    # Create sub-model to get convolutional layer output and model output
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, reg_output = grad_model(input_array)
        # Ensure gradient can be calculated
        reg_output = tf.reduce_sum(reg_output)

    # Calculate gradients
    grads = tape.gradient(reg_output, last_conv_layer_output)

    # Calculate channel importance weights
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    pooled_grads_expanded = tf.expand_dims(tf.expand_dims(pooled_grads, axis=0), axis=0)

    # Calculate heatmap
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = tf.reduce_sum(
        last_conv_layer_output * pooled_grads_expanded,
        axis=-1
    )

    # Normalization processing
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.math.reduce_max(heatmap) + 1e-8  # Avoid division by zero
    heatmap = tf.keras.backend.eval(heatmap)

    # Upsample to original input size (81x81)
    heatmap = cv2.resize(heatmap, (81, 81))
    return heatmap


def find_conv_layer(model):
    """Automatically find the last convolutional layer in the model"""
    conv_layers = []
    for layer in model.layers:
        if 'conv2d' in layer.name.lower() or 'convolution' in layer.name.lower():
            conv_layers.append(layer.name)

    if not conv_layers:
        raise ValueError("No convolutional layers found in the model")

    return conv_layers[-1]  # Return last convolutional layer


if __name__ == "__main__":
    # ===== Configuration Parameters =====
    MODEL_PATH = "./unplotted_RCPM_random_state.h5"  # Replace with your model path
    DATA_PATH = "./eem_features.npy"  # Replace with your eem_features path
    FILE_LIST_PATH = "./processed_files.txt"  # New: path to text file containing filenames
    OUTPUT_DIR = "./heatmaps"  # Output directory

    # ===== Main Program =====
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load model
    print(f"Loading model from {MODEL_PATH}")
    loaded_model = load_model(MODEL_PATH)

    # Find last convolutional layer
    last_conv_name = find_conv_layer(loaded_model)
    print(f"Using convolutional layer: {last_conv_name}")

    # Load data
    print(f"Loading data from {DATA_PATH}")
    X_test = np.load(DATA_PATH)

    print(f"Loading file list from {FILE_LIST_PATH}")
    with open(FILE_LIST_PATH, 'r') as f:
        file_list = [line.strip() for line in f]
    # Ensure file list length matches number of data samples
    assert len(file_list) == len(X_test), "File list length does not match the number of samples"

    # Generate heatmap for each sample
    print(f"Generating heatmaps for {len(X_test)} samples...")
    for i, sample in enumerate(X_test):
        original_filename = os.path.splitext(file_list[i])[0]

        # Prepare input data
        input_array = prepare_input(sample)

        # Generate heatmap
        heatmap = gradram_heatmap(input_array, loaded_model, last_conv_name)

        # Save heatmap
        heatmap = np.clip(heatmap, 0, 1)
        heatmap = np.flipud(heatmap)
        colormap = plt.get_cmap('OrRd')
        colored_heatmap = colormap(heatmap)
        colored_heatmap = (colored_heatmap[:, :, :3] * 255).astype(np.uint8)

        # Create and save image
        img = Image.fromarray(colored_heatmap)
        img.save(os.path.join(OUTPUT_DIR, f'heatmap_{original_filename}.png'))

        flipped_sample = np.flipud(sample.squeeze())
        # Save original data visualization
        plt.imsave(
            os.path.join(OUTPUT_DIR, f'original_{original_filename}.png'),
            flipped_sample,
            cmap='gray'
        )

        # Print progress every 10 samples
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(X_test)} samples")

    print(f"All heatmaps saved to {OUTPUT_DIR}")