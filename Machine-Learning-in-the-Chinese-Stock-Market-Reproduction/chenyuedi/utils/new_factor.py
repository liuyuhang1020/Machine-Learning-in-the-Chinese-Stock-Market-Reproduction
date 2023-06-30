import numpy as np
import pandas as pd

def nextym(ym):
    if ym[-2:] == "12":
        return str(int(ym) + 89)
    return str(int(ym) + 1)
input_path = r"E:\stock\code\data_cyd_v2\data\factor"
output_path = r"C:\Users\chenyuedi\Desktop\new_factor"
#file_lst = ["maxret.csv","mom1m.csv","mom12m.csv","mom6m.csv","mom36m.csv","volatility.csv","zerotrade.csv"]
file_lst=["mom36m.csv"]
for file in file_lst:
    N = pd.read_csv(input_path + "/" + file)
    N['202210'] = np.nan
    for col in N.columns[1:-1]:
        N[col] = N[nextym(col)]
    N.to_csv(output_path + "/" + file,encoding='utf_8_sig',index = False)

