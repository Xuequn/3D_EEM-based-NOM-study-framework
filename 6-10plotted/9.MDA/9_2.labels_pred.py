from tensorflow.keras.models import load_model
from sklearn.decomposition import PCA
import numpy as np
from joblib import load

# Load label scaler - update this path
scaler_labels = load('./scaler_labels_random_state.joblib')
# Path to the pre-trained model - update this path
model_path = "./plotted_RCPM_random_state.h5"
# Load the pre-trained model
model=load_model(model_path)

# Initialize an empty list to store outputs
Y_pred=[]
# Load test data - update this path
X_test=np.load("./image.npy")

for i in range(len(X_test)):
    test_img=X_test[i]  # Get an individual test image
    test_img=test_img[np.newaxis,:, :]  # Add an extra dimension
    test_img=test_img/255  # Normalize the image
    y_pred_normalized = model.predict(test_img)
    y_pred = scaler_labels.inverse_transform(y_pred_normalized)
    Y_pred.append(y_pred)  # Append the output to the list

# Convert list to numpy array
Y_pred = np.array(Y_pred)

# Save predictions - update output path as needed
np.save('./y_pred_random_state.npy', Y_pred)


