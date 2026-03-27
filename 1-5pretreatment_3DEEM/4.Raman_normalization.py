"""
Script to normalize EEM data by Raman integrals.
Requires a sample_raman_integral.xlsx file mapping samples to Raman values.
"""
import pandas as pd
import numpy as np
import os

def matrix_to_excel(filename, ex, em, fl):
    fl = np.insert(fl, 0, ex, 0)
    fl = fl.T
    em = np.insert(em, 0, 0)
    fl = np.insert(fl, 0, em, 0)
    fl = fl.T
    df = pd.DataFrame(fl)
    new_excel_file = os.path.basename(filename)
    save_path = os.path.join("./raman_normalized", new_excel_file) # Update this path to your desired output directory
    df.to_excel(save_path, header=False, index=False)


raman_integrals_path = "./sample_raman_integral.xlsx"# Update this path to your Raman integral file
raman_data = pd.read_excel(raman_integrals_path)
raman_dict = dict(zip(raman_data.iloc[:, 0], raman_data.iloc[:, 1]))

input_directory = "./blank_delete"# Update this path to your input directory containing blank_delete Excel files
for filename in os.listdir(input_directory):
    filepath = os.path.join(input_directory, filename)
    sample = pd.read_excel(filepath, header=None, index_col=None)
    sample = np.array(sample)

    EX = (sample[0])[1:]
    EM = ((sample.T)[0])[1:]
    FL = (((sample[1:]).T)[1:]).T
    raman_value = raman_dict[filename]
    FL = FL / raman_value

    matrix_to_excel(filename, EX, EM, FL)