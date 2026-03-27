"""
Script to extract rows following 'Data Points' marker in Excel files.
"""
import os
import pandas as pd

# Create output folder
output_folder = "./extracted_data" # Update this to your desired output location
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Directory containing the Excel files
directory ="./"  # Update this to your Excel files directory


# Iterate over the files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.xlsx'):
        # Read Excel data
        filepath = os.path.join(directory, filename)
        df = pd.read_excel(filepath)
        data_point_rows = df[df.iloc[:, 0] == 'Data Points'].index

        # Iterate through found rows
        for row_index in data_point_rows:
            # Extract current row and next 100 rows of data
            extracted_data = df.iloc[row_index+1:row_index + 101, :]

            # Construct new Excel filename
            new_excel_file = os.path.splitext(filename)[0] + '.xlsx'
            save_path = os.path.join(output_folder, new_excel_file)

            # Save extracted data to new Excel file
            extracted_data.to_excel(save_path, index=False, header=False)







