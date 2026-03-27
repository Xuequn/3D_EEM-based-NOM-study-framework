import warnings

# Disable all warnings
warnings.filterwarnings("ignore")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt
import scipy
import scipy.io as sio
import sklearn
import umap
import pandas as pd
from umap.parametric_umap import ParametricUMAP
import numpy as np
from mda import *

# Load sample names - update this path
with open("./processed_files.txt", "r") as f:
    sample_names = [line.strip() for line in f.readlines()]

# Font size for all the MDA visualizations shown below
FS = 16

# Number of neighbors in MDA analyses
neighborNum = 5

# Load feature data extracted from images - update this path
testDataFeatures = np.load("./feature_n+1_random_state.npy")
# Load data labels corresponding to images - update this path
Y = np.load("./eem_labels.npy")
# Reshape the target images into vectors so that they can be analyzed by MDA
Y = Y.reshape(Y.shape[0],-1)
# Load output images prediced by the model - update this path
Y_pred = np.load("./y_pred_random_state.npy")
# Reshape the predicted output images into vectors so that they can be analyzed by MDA
Y_pred = Y_pred.reshape(Y_pred.shape[0],-1)

# Create color map for MDA visualization from the target manifold topology
clusterIdx = discoverManifold(Y, neighborNum)
# Compute the outline of the output manifold
clusterIdx_pred = discoverManifold(Y_pred, neighborNum)
# Use the outline of the output manifold to generate the MDA visualization of the features
Yreg = mda(testDataFeatures,clusterIdx_pred)

Yreg_df = pd.DataFrame(Yreg, columns=['MDA1', 'MDA2'])
Yreg_df.insert(0, 'Sample', sample_names)
low_dimention_feature = './feature_random_state_n+1.xlsx' # Update this output path as needed
Yreg_df.to_excel(low_dimention_feature, index=False)

# Plot the MDA results
plt.figure(1, linewidth=0.5, figsize=(5.5, 4.2), dpi=300)
plt.rc('font', family='Arial')
plt.rcParams['font.size'] = 16
plt.scatter(Yreg[:, 0],Yreg[:, 1], c=clusterIdx.T, cmap='jet_r', s=3)
plt.tight_layout()
plt.xlabel("MDA1", labelpad=0.2)
plt.ylabel("MDA2", labelpad=0.1)
plt.savefig('./feature_random_state_n+1.png')# Update this output path as needed
plt.show()
