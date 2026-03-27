"""
Script to extract intermediate features from a pre-trained CNN model.
Extracts features from the 'conv2d_n' layer and applies PCA for dimensionality reduction.
"""
from tensorflow.keras.models import load_model
from sklearn.decomposition import PCA
from joblib import load
import numpy as np
import tensorflow as tf

# Path to the pre-trained model - update this path
model_path = "./unplotted_RCPM_random_state.h5"
model = load_model(model_path)

# Load feature scaler - update this path
scaler_features = load("./scalers_feature_random_state.joblib")
feature_max = scaler_features['feature_max']

# Load test data - update this path
X_test=np.load("./eem_features.npy")
X_test = X_test.astype('float32') / feature_max

# Get intermediate layer output
interlayer_output=model.get_layer('conv2d_n').output
inter_model = tf.keras.Model(inputs=model.input, outputs=interlayer_output)

# Process in batches
batch_size = 32
inter_out = []
for i in range(0, len(X_test), batch_size):
    batch = X_test[i:i+batch_size]
    batch_out = inter_model.predict(batch)
    inter_out.append(batch_out)

inter_out = np.vstack(inter_out)
print("Feature shape:", inter_out.shape)

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
np.save('./feature_n+1_random_state.npy', inter_out)


