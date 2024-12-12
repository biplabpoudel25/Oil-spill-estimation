import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Function to calculate average RGB value of an image
def calculate_avg_rgb(image_path):
    img = cv2.imread(image_path)
    avg_color = np.mean(img, axis=(0, 1))
    # Convert BGR to RGB mean
    avg_rgb = np.mean(avg_color)
    return avg_rgb


# List of CSV files and their corresponding labels and colors
csv_files = [
    {'file': r"CSV Files\11_13_2023\combined_data.csv", 'label': 'Nov NACO 2023', 'color': 'blue', 'marker': '*'},
    {'file': r"CSV Files\12_11_2023\20231211 AF\20231211 AF.csv", 'label': 'Dec NACO 2023', 'color': 'red', 'marker': 'o'},
    {'file': r"CSV Files\2024_06_27_ANCO\Full size img ANCO\image_data.csv", 'label': 'June ANCO 2024', 'color': 'green', 'marker': '+'},
    {'file': r"CSV Files\2024_07_21_ANCO\MB_Volumetric\Full Size img ANCO\image_data.csv", 'label': 'July ANCO 2024', 'color': 'purple', 'marker': 'x'}
]

# Create the plot
plt.figure(figsize=(8, 6))

# Process each CSV file
for csv_info in csv_files:
    # Read CSV file
    df = pd.read_csv(csv_info['file'])

    # Group by concentration and calculate mean RGB for each concentration
    concentration_groups = df.groupby('label')
    avg_rgb_values = []
    concentrations = []

    for concentration, group in concentration_groups:
        rgb_values = [calculate_avg_rgb(path) for path in group['datapath']]
        avg_rgb_values.append(np.mean(rgb_values))
        concentrations.append(float(concentration))

    # Sort the data by concentration for proper line plotting
    sort_idx = np.argsort(concentrations)
    concentrations = np.array(concentrations)[sort_idx]
    avg_rgb_values = np.array(avg_rgb_values)[sort_idx]

    plt.plot(concentrations, avg_rgb_values,
             color=csv_info['color'],
             alpha=0.7,
             zorder=1)

    plt.scatter(concentrations, avg_rgb_values,
                color=csv_info['color'],
                marker=csv_info['marker'],
                label=csv_info['label'],
                s=15,
                zorder=2)

# Customize the plot
plt.xlabel('Concentration (mg/L)')
plt.ylabel('Average RGB value')
plt.legend()
plt.grid(False)
plt.tight_layout()
# plt.savefig('rgb_concentration_plot.png', dpi=1200, bbox_inches='tight')
plt.show()