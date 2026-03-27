"""
Script for hierarchical clustering analysis of low-dimensional features.
Performs clustering with multiple distance thresholds and visualizes results.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

# Load data - update this path to your feature file
data = pd.read_excel("./feature_random_state_n+1.xlsx")
filename = data["Sample"].values
feature_0 = data['0'].values
feature_1 = data['1'].values
feature = np.array(list(zip(feature_0, feature_1)))

# Use hierarchical clustering
linked = linkage(feature, method='ward')

# Define different thresholds to test classification results
thresholds = [10, 20, 30, 50, 75, 100]

# Cluster and visualize for each threshold
for threshold in thresholds:
    print(f"Processing with threshold: {threshold}")

    # Cluster based on threshold
    clusters = fcluster(linked, t=threshold, criterion='distance')

    # Create DataFrame to save image paths and clustering results
    data = {
        'filename': filename,
        'Cluster': clusters,
    }
    df = pd.DataFrame(data)

    # Save DataFrame to Excel file
    output_file_path = fr'./cluster_random_state_n+1/threshold_{threshold}.xlsx' # update this path as needed
    df.to_excel(output_file_path, index=False)

    print(f'Data and clusters with threshold {threshold} have been saved to {output_file_path}')

    # Plot dendrogram
    plt.figure(linewidth=0.5, figsize=(5.5, 4.2), dpi=300)
    plt.rc('font', family='Arial')
    plt.rcParams['font.size'] = 16
    plt.tight_layout()
    dendrogram(linked)
    plt.ylabel('Distance', labelpad=0.2)
    dendrogram_output_path = fr'./cluster_random_state_n+1/feature.png'
    plt.savefig(dendrogram_output_path)

    # Visualize clustering results
    plt.figure(linewidth=0.5, figsize=(8, 5.3), dpi=300)
    plt.rc('font', family='Arial')
    plt.rcParams['font.size'] = 16
    plt.tight_layout()
    plt.scatter(feature[:, 0], feature[:, 1], c=clusters, cmap='jet')
    plt.colorbar(ticks=clusters)
    plt.title(f'Hierarchical Clustering Results (Threshold={threshold})')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    scatter_output_path = fr"./cluster_random_state_n+1/threshold_{threshold}.png"
    plt.savefig(scatter_output_path)

    print(f'Visualizations with threshold {threshold} have been saved to {dendrogram_output_path} ')