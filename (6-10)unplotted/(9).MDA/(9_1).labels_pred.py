from tensorflow.keras.models import load_model
from joblib import load
import numpy as np
import tensorflow as tf

# Load scalers - update these paths to your actual scaler files
scaler_labels = load("./scaler_labels_random_state.joblib")
scaler_features = load("./scalers_feature_random_state.joblib")
feature_max = scaler_features['feature_max']

# Path to the pre-trained model - update this path
model_path = "./unplotted_RCPM_6.h5"
# Load the pre-trained model
model = load_model(model_path)

# Initialize an empty list to store outputs
Y_pred = []
# Load test data - update this path to your test data
X = np.load("./eem_features.npy")
X_scaled = X.astype('float32') / np.max(X)

Y_pred_normalized = model.predict(X_scaled)

Y_pred = scaler_labels.inverse_transform(Y_pred_normalized)

np.save("./y_pred_random_state.npy", Y_pred)


