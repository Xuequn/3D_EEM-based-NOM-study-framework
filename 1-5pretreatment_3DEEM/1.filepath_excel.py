"""
Script to collect image file paths and save to Excel.
Update folder_path variable with your image directory.
"""
import os
import pandas as pd

folder_path = r"PATH_TO_YOUR_IMAGE_FOLDER"

file_paths = []
for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    if os.path.isfile(file_path):
        file_paths.append(file_path)

# Create a DataFrame and store file paths in a column
df = pd.DataFrame({'file_path': file_paths})
# Save DataFrame to Excel file
df.to_excel(r"image_paths.xlsx", index=False) # Saved in current directory