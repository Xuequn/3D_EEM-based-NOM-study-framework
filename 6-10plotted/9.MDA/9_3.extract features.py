"""
Script to extract intermediate features from a pre-trained CNN image model.
Extracts features from the 'conv2d_n' layer and applies PCA for dimensionality reduction.
"""
from tensorflow.keras.models import load_model
from sklearn.decomposition import PCA
import numpy as np
import tensorflow as tf

# Path to the pre-trained model - update this path
model_path="./plotted_RCPM_random_state.h5"
model = load_model(model_path)

# Load test data - update this path
X_test=np.load("./image.npy")

# Get intermediate layer output
interlayer_output=model.get_layer('conv2d_n').output
inter_model = tf.keras.Model(inputs=model.input, outputs=interlayer_output)

# Extract features for each image
inter_out=[]
for i in range(len(X_test)):
    test_img=X_test[i]
    test_img=test_img[np.newaxis,:, :]
    test_img=test_img/255
    test_out=inter_model.predict(test_img)
    test_out=np.squeeze(test_out)
    inter_out.append(test_out)

inter_out=np.array(inter_out)
print(inter_out.shape)

# Flatten features
n1, h1, w1, c1 = inter_out.shape
inter_out = inter_out.reshape(-1, h1*w1*c1)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=100, svd_solver='arpack')
inter_out = pca.fit_transform(inter_out)
print(inter_out.shape)

# Save extracted features - update output path as needed
np.save('./MDA_random_state/feature_n+1.npy', inter_out)


