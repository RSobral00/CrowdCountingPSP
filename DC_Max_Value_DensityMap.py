#Code to get the maximum vlaue of the density maps 

import os
import scipy.io as io
import numpy as np

def find_top_max_values_across_heatmaps(folder_path, top_n=10):
    all_max_values = []

    for name in os.listdir(folder_path):
       
        file_path = os.path.join(folder_path,name)
        heatmap = io.loadmat(file_path)["heatmap"]
        max_value = np.max(heatmap)
        print(max_value)
        all_max_values.append(max_value)

    top_max_values = sorted(all_max_values, reverse=True)[:top_n]
    return top_max_values

train_folder_heatmaps = os.path.expanduser("~/Desktop/Tese/DroneCrowd/train_data/heatmaps_train/")
val_folder_heatmaps = os.path.expanduser("~/Desktop/Tese/DroneCrowd/val_data/heatmaps_val/")
test_folder_heatmaps = os.path.expanduser("~/Desktop/Tese/DroneCrowd/test_data/heatmaps_test/")

list_folders = [train_folder_heatmaps, val_folder_heatmaps, test_folder_heatmaps]

top_n = 10

overall_top_max_values = []

for folder_path in list_folders:
    top_max_values = find_top_max_values_across_heatmaps(folder_path, top_n)
    overall_top_max_values.extend(top_max_values)

overall_top_max_values = sorted(overall_top_max_values, reverse=True)[:top_n]

print(f"Overall top {top_n} max values: {overall_top_max_values}")


#Overall top 10 max values: [0.04163233, 0.041627534, 0.041607574, 0.04139385, 0.04139385, 0.041380126, 0.041326262, 0.041326262, 0.041320775, 0.041243434]