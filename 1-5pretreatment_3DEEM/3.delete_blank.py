"""
Script to subtract blank values from sample data.
Requires a sample_blank.xlsx file with sample and corresponding blank path.
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
    save_path = os.path.join("./blank_delete", new_excel_file) # Update this path to your desired output directory
    df.to_excel(save_path, header=False, index=False)

data = pd.read_excel("./sample_blank.xlsx")# Update this path to your sample_blank.xlsx file location

extract_dir = "./extract"# Update these paths to your extract directory
file_paths = data["sample"].apply(lambda x: os.path.join(extract_dir, x)).values
blank_paths = data["blank"].apply(lambda x: os.path.join(extract_dir, x)).values

for file_path, blank_path in zip(file_paths, blank_paths):
    sample = pd.read_excel(file_path, header=None, index_col=None)
    sample = np.array(sample)
    blank = pd.read_excel(blank_path, header=None, index_col=None)
    blank = np.array(blank)

    EX = sample[0, 1:]
    EM = sample[1:, 0]
    FL = sample[1:, 1:]


    EX_blank = blank[0, 1:]
    EM_blank = blank[1:, 0]
    FL_blank = blank[1:, 1:]

    blank_mapping = {}
    for a, em_blank in enumerate(EM_blank):
        for b, ex_blank in enumerate(EX_blank):
            blank_mapping[(em_blank, ex_blank)] = FL_blank[a, b]

    for i, em in enumerate(EM):
        for j, ex in enumerate(EX):
            if (em, ex) in blank_mapping:
                FL[i, j] -= blank_mapping[(em, ex)]
            else:
                FL[i, j] -= 0

            if FL[i, j] < 0:
                FL[i, j] = 0

    matrix_to_excel(file_path, EX, EM, FL)