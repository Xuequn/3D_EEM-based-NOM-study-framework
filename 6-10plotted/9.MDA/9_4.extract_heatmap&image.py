import numpy as np
import  os
from skimage import io, transform
from sklearn.decomposition import PCA

# Directory containing heatmap images - update this path
directory = './heatmaps_random_state'
target_size = (560, 480)

images = []
image_paths = []

for filename in os.listdir(directory):
    image_path = os.path.join(directory, filename)
    image = io.imread(image_path)

    # Resize image to target size
    image_resized = transform.resize(image, target_size, anti_aliasing=True)

    # Flatten image to 1D array and normalize
    image_flattened = image_resized.flatten() / 255.0

    # Add processed image to list
    images.append(image_flattened)
    image_paths.append(image_path)

# Convert list to numpy array
images_np = np.array(images)

# Apply PCA for dimensionality reduction (optional)
pca = PCA(n_components=100,svd_solver='arpack')
images_pca = pca.fit_transform(images_np)

# Save features - update output path as needed
np.save('./feature_heatmaps_random_state.npy', images_pca)
