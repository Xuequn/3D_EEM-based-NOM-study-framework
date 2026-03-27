import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load data from Excel file - update this path to your data file
data = pd.read_excel("./image_path_label.xlsx")

image_paths = data["image_path"].values
labels = data["label"].values
sample_names = data["image_path"].values

# Load and process images
images = []
for image_path in image_paths:
    image = load_img(image_path, target_size=(560, 480))
    image = img_to_array(image)
    images.append(image)

images = np.array(images)

np.save('image.npy', images)
np.save('label.npy', labels)
np.save('sample_names.npy', sample_names)