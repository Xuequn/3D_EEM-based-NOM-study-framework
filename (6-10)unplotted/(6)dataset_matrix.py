"""
Script to process EEM data and create standardized dataset with labels.
Aligns fluorescence data to standard wavelength grid and integrates labels.
"""
import numpy as np
import pandas as pd
import os

# Define standard wavelength range
EX_START, EX_END, EX_STEP = 200, 600, 5
EM_START, EM_END, EM_STEP = 200, 600, 5

# Generate standard wavelength grid
ex_standard = np.arange(EX_START, EX_END + EX_STEP, EX_STEP)
em_standard = np.arange(EM_START, EM_END + EM_STEP, EM_STEP)


def process_fluorescence_excel(file_path):
    """Process a single fluorescence Excel file and align to standard wavelength grid"""
    # Read Excel file (without auto-parsing headers)
    df = pd.read_excel(file_path, header=None)

    # Extract excitation wavelengths (first row, skip first cell)
    ex_raw = df.iloc[1:, 0].values.astype(float)
    # Extract emission wavelengths (first column, skip first cell)
    em_raw = df.iloc[0, 1:].values.astype(float)
    # Extract fluorescence intensity matrix (starting from cell B2)
    intensity_matrix = df.iloc[1:, 1:].values.astype(float)
    intensity_matrix = intensity_matrix.T

    # Create aligned matrix (initialized to 0)
    aligned_matrix = np.zeros((len(em_standard), len(ex_standard)))

    # Find valid data points within standard range
    ex_valid_mask = (ex_raw >= EX_START) & (ex_raw <= EX_END)
    em_valid_mask = (em_raw >= EM_START) & (em_raw <= EM_END)

    # Get subset of valid data
    valid_ex = ex_raw[ex_valid_mask]
    valid_em = em_raw[em_valid_mask]
    valid_data = intensity_matrix[np.ix_(em_valid_mask, ex_valid_mask)]

    # Calculate positions of valid data in standard grid
    ex_indices = ((valid_ex - EX_START) / EX_STEP).astype(int)
    em_indices = ((valid_em - EM_START) / EM_STEP).astype(int)

    # Map valid data to standard grid
    aligned_matrix[np.ix_(em_indices, ex_indices)] = valid_data

    return aligned_matrix


def load_dataset_with_labels(eem_dir, label_file_path, filename_col='filename', label_col='label'):
    """
       Load EEM dataset and integrate labels

       Parameters:
           eem_dir (str): Directory containing EEM Excel files
           label_file_path (str): Path to Excel file containing filenames and labels
           filename_col (str): Column name for filenames
           label_col (str): Column name for labels

       Returns:
           X (np.ndarray): Feature tensor (n_samples, 81, 81, 1)
           y (np.ndarray): Label array (n_samples,)
           file_list (list): List of successfully processed filenames
       """
    # Read label file
    label_df = pd.read_excel(label_file_path)

    # Create dictionary mapping filenames to labels
    label_dict = dict(zip(label_df[filename_col], label_df[label_col]))

    all_samples = []
    all_labels = []
    matched_files = []
    unmatched_files = []

    # Iterate through all files in EEM directory
    for filename in os.listdir(eem_dir):
        if filename.endswith(('.xlsx', '.xls')):
            file_path = os.path.join(eem_dir, filename)

            # Check if filename exists in label dictionary
            if filename in label_dict:
                try:
                    # Process EEM data
                    processed_data = process_fluorescence_excel(file_path)
                    all_samples.append(processed_data)

                    # Get corresponding label
                    label_value = label_dict[filename]
                    all_labels.append(label_value)

                    matched_files.append(filename)
                    print(f"Successfully matched and processed: {filename} -> Label: {label_value}")

                except Exception as e:
                    print(f"Failed to process {filename}: {str(e)}")
            else:
                unmatched_files.append(filename)
                print(f"Warning: {filename} not found in label file, skipped")

    # Convert to NumPy arrays
    X = np.array(all_samples)
    y = np.array(all_labels)

    # Add channel dimension (n_samples, height, width, channels)
    X = X[..., np.newaxis]

    # Print statistics
    print("\n" + "=" * 50)
    print(f"Successfully processed files: {len(matched_files)}")
    print(f"Unmatched files: {len(unmatched_files)}")
    print(f"Feature data shape: {X.shape}")
    print(f"Label data shape: {y.shape}")
    print(f"Label range: [{np.min(y):.4f}, {np.max(y):.4f}]")
    print(f"Data types: X={X.dtype}, y={y.dtype}")
    print("=" * 50)

    return X, y, matched_files


if __name__ == "__main__":
    # Configure paths - update these to your actual directories
    EEM_DIR = "./eem_data"  # Directory containing EEM Excel files
    LABEL_FILE = "./labels.xlsx" # Label Excel file

    # Load dataset
    X, y, file_list = load_dataset_with_labels(
        eem_dir=EEM_DIR,
        label_file_path=LABEL_FILE,
        filename_col='filename',
        label_col='label'
    )

    # Save dataset
    np.save("eem_features.npy", X)
    np.save("eem_labels.npy", y)


    # Save filename list
    with open("processed_files.txt", "w") as f:
        f.write("\n".join(file_list))
    print("Processed file list saved as processed_files.txt")