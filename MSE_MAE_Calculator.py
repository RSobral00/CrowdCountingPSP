# Rúben Sobral - Model Evaluation DroneCrowd Metrics

import os
import pandas as pd
import numpy as np 
import math 

csv_path = os.path.expanduser("~/Desktop/Tese/CrowdCountingUAV/Results/Test_Info_ARCN.csv")

data = pd.read_csv(csv_path,index_col=False)

# Calculate Root Mean Squared Error (RMSE)
data["MSE_P1"] = (data['N_Gt'] - data['N_Pred'])**2
rmse = math.sqrt((data["MSE_P1"].mean()))

# Calculate Mean Absolute Error (MAE)
data["MAE_P1"] = abs(data['N_Gt'] - data['N_Pred'])
mae = data["MAE_P1"].mean()

# Calculate Mean Absolute Percentage Error (MAPE)
data["MAPE_P1"] = (abs(data['N_Gt'] - data['N_Pred']) / data['N_Gt']) * 100
mape = data["MAPE_P1"].mean()

print(data)

print("Max GT",data["N_Gt"].max())
print("Min GT",data["N_Gt"].min())
print("Max Pred",data["N_Pred"].max())
print("Min Pred",data["N_Pred"].min())

print("MSE (RMSE): {:.2f}".format(rmse))
print("MAE: {:.2f}".format(mae))
print("MAPE: {:.2f}%".format(mape))