# 3D_EEM-based-NOM-study-framework
Project Overview
This project develops a Convolutional Neural Network (CNN)-based prediction model (RCPM) that directly predicts organic matter redox capacity from 3D excitation-emission matrix (EEM) fluorescence spectra. The framework not only achieves high-precision predictions but also provides interpretable model decisions through Grad-RAM and MDA visualization techniques. Hierarchical clustering based on MDA features enables intrinsic sample classification by spectral patterns and redox functionality.



Key Features
Data Preprocessing - Complete pipeline from raw instrument data to standardized fluorescence matrices

Dual Training Modes:

Raw Data Mode: Direct training using preprocessed fluorescence matrices

Image Mode: Training on visualized fluorescence data as images

Model Interpretability - Grad-RAM and MDA techniques to reveal decision mechanisms

Feature Analysis & Clustering - Hierarchical clustering based on MDA-reduced features

Transfer Learning - Fine-tuning pre-trained models on new data



Project Structure
Project Directory/
├── 1.filepath_excel.py              # Collect image file paths
├── 2.extract_data_point.py          # Extract data points from Excel
├── 3.delete_blank.py               # Blank value subtraction
├── 4.Raman_normalization.py        # Raman normalization
├── 5.scatter_remove.py             # Scatter region removal & interpolation
├── (6)dataset_matrix.py            # Create standardized dataset (raw data mode)
├── (7).pure_data_CNN.py            # Raw data CNN training
├── (8).gradram_pure_data.py        # Raw data Grad-RAM visualization
├── 6.excel_to_fig.py               # Data to image conversion
├── 7.Plotted_predict.py            # Image mode CNN training
├── 8.grad_ram.py                   # Image mode Grad-RAM visualization
├── 9_*.py series                   # Feature extraction & MDA analysis
├── 10.cluster_feature.py           # Hierarchical clustering analysis
├── transfer_learning.py            # Transfer learning
├── mda.py                          # MDA algorithm implementation
└── Configuration & Data/
    ├── sample_blank.xlsx           # Sample-blank reference table
    ├── sample_raman_integral.xlsx  # Raman integral values
    ├── image_path_label.xlsx       # Image path-label mapping
    └── Various output directories/
        ├── extracted_data/         # Extracted data
        ├── blank_delete/           # Blank-corrected data
        ├── raman_normalized/       # Raman-normalized data
        ├── interpolated/           # Interpolated data
        ├── images/                 # Generated EEM images
        ├── heatmaps/               # Grad-RAM heatmaps
        └── cluster_*/              # Clustering results




Environment Setup
This project requires two separate Conda environments due to dependency conflicts:

Environment 1: RCPM_PREDICT (Main Environment)

Environment 2: MDA (MDA Analysis Only)



Usage Workflow

Phase 1: Data Preprocessing (Scripts 1-5)
Collect file paths (1.filepath_excel.py)

Extract data points (2.extract_data_point.py)

Subtract blank values (3.delete_blank.py) - Requires sample_blank.xlsx

Raman normalization (4.Raman_normalization.py) - Requires sample_raman_integral.xlsx

Scatter removal & interpolation (5.scatter_remove.py)



Phase 2A: Raw Data Mode Training
Create standardized dataset ((6)dataset_matrix.py)

CNN model training ((7).pure_data_CNN.py)

Grad-RAM visualization ((8).gradram_pure_data.py)

Data from model preparation for MDA ((9_1).labels_pred.py; (9_2).extract features.py; )

MDA feature analysis ((9_3).MDA.py)- Requires switching to MDA environment

Hierarchical clustering (10.cluster_feature.py)



Phase 2B:Image Mode Training
Data to image conversion (6.excel_to_fig.py)

Image mode CNN training (7.Plotted_predict.py)

Grad-RAM visualization (8.grad_ram.py)

Data from model preparation  for MDA(9_1.labels_to_npy.py; 9_2.labels_pred.py; 9_3.extract features.py)

Image preparation for MDA (9_4.extract_heatmap&image.py)

MDA feature analysis (9_5.MDA_plotted.py) - Requires switching to MDA environment

Hierarchical clustering (10.cluster_feature.py)



Transfer Learning
transfer_learning.py - Fine-tune pre-trained models



Configuration
Each script contains path variables that need updating:

Data paths: Point to your EEM data directories

Label files: Excel files with sample filenames and corresponding reducing capacity labels

Output directories: Results storage locations for each processing step

Auxiliary files: sample_blank.xlsx, sample_raman_integral.xlsx.
