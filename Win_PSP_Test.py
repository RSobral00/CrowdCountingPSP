#Get test results to # IMPORTANTE- RESULTADOS PARA A TESE 


import os 
import pandas as pd 
import scipy.io as io 
import numpy as np 
import math 


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

test_Gt_info = os.path.expanduser("~/Desktop/Tese/PSP_Dataset/test_info/PSP_GT_Info.txt")
predictions = os.path.expanduser("~/Desktop/Tese/CrowdCountingPSP/Results/PSP_DC_Soft_CSRNETP_Predictions")

test_info = pd.read_csv(test_Gt_info, sep=",", header=None)
test_info.columns = ["Img", "N_Gt"]


test_info['Img_Numbers'] = test_info['Img'].str.extract('(\d+)').astype(int)

test_info_sorted = test_info.sort_values(by=["Img_Numbers", "Img"])

test_info_sorted = test_info_sorted.drop(columns=["Img_Numbers"])

test_info_sorted = test_info_sorted.reset_index(drop=True)


def compute_density_map_sum(img_name):
    mat_file = os.path.join(predictions, img_name.split('.')[0] + "_prediction.mat")
    density_map = io.loadmat(mat_file)["heatmap"]
    return np.sum(density_map)


test_info_sorted['N_Pred'] = test_info_sorted['Img'].apply(compute_density_map_sum)

test_info = test_info_sorted

test_info["MSE_Solo"] = (test_info['N_Gt'] - test_info['N_Pred'])**2
test_info["MAE_Solo"] = abs(test_info['N_Gt'] - test_info['N_Pred'])
test_info["MAPE_Solo"] = (np.abs((test_info['N_Gt'] - test_info['N_Pred']) / test_info['N_Gt']))*100 


print(test_info)

# Calculate stds for each metric
std_rmse_mse = np.std(test_info["MSE_Solo"], ddof=1)


std_mae = np.std(test_info["MAE_Solo"], ddof=1)
std_mape = np.std(test_info["MAPE_Solo"], ddof=1)


print("RMSE-MSE:",math.sqrt(np.mean(test_info["MSE_Solo"])))
print("MAE:",np.mean(test_info["MAE_Solo"]))
print("MAPE:",np.mean(test_info["MAPE_Solo"]))





print("RMSE-MSE Std:", np.sqrt(std_rmse_mse))
print("MAE Std:", std_mae)
print("MAPE Std:", std_mape)

mean_rmse_mse = math.sqrt(np.mean(test_info_sorted["MSE_Solo"]))
mean_mae = np.mean(test_info_sorted["MAE_Solo"])
mean_mape = np.mean(test_info_sorted["MAPE_Solo"])
# Calculate coefficient of variation for each metric without percentage
cv_rmse_mse = np.sqrt(std_rmse_mse) / mean_rmse_mse
cv_mae = std_mae / mean_mae
cv_mape = std_mape / mean_mape

# Print coefficient of variation without percentage
print("RMSE-MSE CV:", cv_rmse_mse)
print("MAE CV:", cv_mae)
print("MAPE CV:", cv_mape)



save_path = os.path.expanduser("~/Desktop/Tese/CrowdCountingPSP/Results/PSP_DC_Soft_CSRNETP_Predictions.csv")

test_info_sorted.to_csv(save_path, index=False, header=True)
