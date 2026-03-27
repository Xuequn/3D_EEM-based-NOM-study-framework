import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Directory containing the pretreated Excel files - update this path
directory = "./pretreated"

for filename in os.listdir(directory):
    filepath = os.path.join(directory, filename)
    data = pd.read_excel(filepath, header=None)

    x = data.iloc[1:, 0]
    y = data.iloc[0, 1:]

    z = data.iloc[1:, 1:].values
    z = z.astype(float)
    z = np.transpose(z)
    z = np.flipud(z)

    X, Y = np.meshgrid(x, y)

    width_px = 560
    height_px = 480

    fig, ax = plt.subplots(figsize=(width_px / 100, height_px / 100), dpi=100)

    im = ax.imshow(z, cmap='viridis', extent=[x.min(), x.max(), y.min(), y.max()], aspect=1)

    colorbar = plt.colorbar(im, ax=ax, ticks=np.linspace(z.min(), z.max(), 6))

    ax.set_xlim(200, 600)
    ax.set_ylim(200, 600)

    ax.axis('off')

    ax.set_xticks([])
    ax.set_yticks([])

    image_filename = os.path.splitext(filename)[0] + '.png'
    save_path = os.path.join("./images", image_filename)  # Update this output directory path
    plt.savefig(save_path)
    plt.close()

