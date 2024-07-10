import os
import numpy as np
import scipy.io as io
from skimage.metrics import structural_similarity as ssim

# Paths to the folders
path_dm_pred = save_path = os.path.expanduser("~/Desktop/Tese/CrowdCountingPSP/Results/PSP_DC_Soft_CSRNETP_Predictions")
path_dm_gt = os.path.expanduser("~/Desktop/Tese/PSP_Dataset/test_labels")

ssim_values = []

for filename in os.listdir(path_dm_gt):
    if filename.endswith('.mat'):
        pred_filename = filename.replace('.mat', '_prediction.mat')
        pred_path = os.path.join(path_dm_pred, pred_filename)
        gt_path = os.path.join(path_dm_gt, filename)

        if os.path.exists(pred_path):
            # Load ground truth density map
            gt_density_map = io.loadmat(gt_path)["heatmap"]
            gt_density_map = np.array(gt_density_map, dtype=np.float32)
            gt_data_range = gt_density_map.max() - gt_density_map.min()
            
            # Load predicted density map and scale it
            pred_density_map = io.loadmat(pred_path)["heatmap"]
            pred_density_map = np.array(pred_density_map, dtype=np.float32) 
            pred_density_map = np.squeeze(pred_density_map, axis=(0, 3))
          
            # Compute SSIM between the two density maps
            ssim_value, _ = ssim(gt_density_map, pred_density_map, data_range=gt_data_range, full=True)
            ssim_values.append(ssim_value)

# Calculate mean and standard deviation of SSIM values
mean_ssim = np.mean(ssim_values)
std_ssim = np.std(ssim_values)


# Print the mean and standard deviation of SSIM values
print(f"Mean SSIM: {mean_ssim}")
print(f"STD of SSIM: {std_ssim}")
print("End")
