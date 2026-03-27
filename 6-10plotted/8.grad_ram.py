"""
Generate Grad-CAM heatmaps for EEM image models.
Visualizes which regions of the EEM images are important for model predictions.
"""
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from matplotlib import cm
from PIL import Image
import matplotlib.pyplot as plt
import os
import pandas as pd


def prepare_input(image_path):
    """Prepare RGB image input (560x480x3)"""
    image = load_img(image_path, target_size=(560, 480))
    image = img_to_array(image)
    image = image.astype('float32') / 255.0
    return np.expand_dims(image, axis=0)   # Add batch dimension

def gradram_heatmap(input_array, model, last_conv_layer_name):
    """Generate Grad-CAM heatmap for RGB image model"""
    # Create sub-model to get convolutional layer output and model output
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, reg_output = grad_model(input_array)
        reg_output = tf.reduce_sum(reg_output)

    # Calculate gradients
    grads = tape.gradient(reg_output, last_conv_layer_output)

    # Calculate channel importance weights
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Calculate heatmap
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = tf.reduce_sum(
        last_conv_layer_output * pooled_grads,
        axis=-1
    )

    # Normalization processing
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val != 0:   # Avoid division by zero
        heatmap /= max_val
    heatmap = heatmap.numpy()

    # Upsample to original input size (560x480)
    heatmap = cv2.resize(heatmap, (560, 480))  # OpenCV uses (width, height)
    return heatmap


def find_conv_layer(model):
    """Automatically find the last convolutional layer in the model"""
    conv_layers = []
    for layer in model.layers:
        if 'conv2d' in layer.name.lower():
            conv_layers.append(layer.name)

    if not conv_layers:
        raise ValueError("No convolutional layers found in the model")

    return conv_layers[-1]  # Return last convolutional layer


def overlay_heatmap(original_img, heatmap, alpha=0.5):
    """Overlay heatmap onto original image"""
    # Convert heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Resize heatmap to match original image dimensions
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))

    # Overlay images
    overlayed = cv2.addWeighted(original_img, 1 - alpha, heatmap, alpha, 0)
    return overlayed


if __name__ == "__main__":
    # ===== Configuration Parameters =====
    MODEL_PATH = "./plotted_RCPM_random_state.h5"  # Replace with your model path
    DATA_EXCEL = "./image_path_label.xlsx"  # Excel file containing image paths and corresponding labels
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
    print(f"Loading data from {DATA_EXCEL}")
    df = pd.read_excel(DATA_EXCEL)
    image_paths = df["image_path"].values

    print(f"Generating heatmaps for {len(image_paths)} images...")
    for i, img_path in enumerate(image_paths):
            # Get filename (without extension)
            filename = os.path.splitext(os.path.basename(img_path))[0]

            # Prepare input data
            input_array = prepare_input(img_path)

            # Generate heatmap
            heatmap = gradram_heatmap(input_array, loaded_model, last_conv_name)

            plt.imsave(
                os.path.join(OUTPUT_DIR, f'heatmap_{filename}.png'),
                heatmap,
                cmap='OrRd'
            )

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(image_paths)} images")

    print(f"All heatmaps saved to {OUTPUT_DIR}")