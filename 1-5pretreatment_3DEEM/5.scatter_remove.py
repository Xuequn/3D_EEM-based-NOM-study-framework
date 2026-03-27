"""
Script to remove scatter regions and interpolate missing values in EEM data.
Removes specific regions based on EX/EM relationships and interpolates missing values.
"""
import pandas as pd
import numpy as np
import os
from scipy.interpolate import LinearNDInterpolator, griddata

def matrix_to_excel(filename, ex, em, fl):
    fl = np.insert(fl, 0, ex, 0)
    fl = fl.T
    em = np.insert(em, 0, 0)
    fl = np.insert(fl, 0, em, 0)
    fl = fl.T
    df = pd.DataFrame(fl)
    new_excel_file = os.path.splitext(filename)[0] + '.xlsx'
    save_path = os.path.join("./interpolated", new_excel_file) # Update this path to your desired output directory
    df.to_excel(save_path, header=False, index=False)

directory = "./raman_normalized" # Update this path to your input directory containing Raman normalized Excel files
for filename in os.listdir(directory):
    filepath = os.path.join(directory, filename)
    # Read Excel data
    sample = pd.read_excel(filepath, header=None, index_col=None)
    sample = np.array(sample)

    EX = sample[0, 1:]
    EM = sample[1:, 0]
    FL = sample[1:, 1:].astype(float)

    FL = FL.T
    for i in range(len(FL)):
        for j in range(len(FL[i])):
            ex = EX[i]
            em = EM[j]
            if (ex >= em - 25 and ex <= em + 15):
                FL[i, j] = np.nan

            if (ex >= 0.75 * em + 40 and ex <= 0.75 * em + 55):
                FL[i, j] = np.nan

            if (ex >= 0.5 * em - 10 and ex <= 0.5 * em + 5):
                FL[i, j] = np.nan

            if (ex >= 0.4 * em + 20 and ex <= 0.4 * em + 40):
                FL[i, j] = np.nan

    FL = FL.T

    ex_mesh, em_mesh = np.meshgrid(EX, EM)

    points = np.column_stack((ex_mesh.flatten(), em_mesh.flatten()))
    values = FL.flatten()

    valid_points = points[~np.isnan(values)]
    valid_values = values[~np.isnan(values)]

    interpolator = LinearNDInterpolator(valid_points, valid_values, fill_value=np.nan)
    interpolated_values = interpolator(points).reshape(FL.shape)

    FL[np.isnan(FL)] = interpolated_values[np.isnan(FL)]

    values = FL.flatten()

    valid_points = points[~np.isnan(values)]
    valid_values = values[~np.isnan(values)]

    nan_mask = np.isnan(FL)
    nan_points = np.column_stack((ex_mesh[nan_mask], em_mesh[nan_mask]))

    nan_values = griddata(valid_points, valid_values, nan_points, method='nearest')

    FL[nan_mask] = nan_values

    matrix_to_excel(filename, EX, EM, FL)