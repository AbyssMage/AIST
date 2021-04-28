from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from utils import *

# 0, 5, 6, 7, 71, 21, 22, 23, 24, 27, 31
# 66, 68, 70, 7, 71, 42, 43, 48, 22, 23, 24
# 66, 68, 70, 7, 71, 42, 43, 22, 24, 28
# 70, 7, 71, 42, 22, 24, 25, 26, 28, 29

target_region = [70, 42, 22, 24, 25, 26, 28, 29]
target_cat = [7]  # (starts from 0)
time_step = 20
f = open('C:/Users/Yeasir Rayhan Prince/PycharmProjects/AIST/ha_mse.txt', 'a')
for reg in target_region:
    for cat in target_cat:
        loaded_data = torch.from_numpy(np.loadtxt("data/com_crime/r_" + str(reg) + ".txt", dtype=np.int)).T
        loaded_data = loaded_data[:, cat:cat + 1]
        x, y, x_daily, x_weekly = create_inout_sequences(loaded_data, 120)

        # Divide into train_test data
        train_x_size = int(x.shape[0] * .67)  # batch_size
        test_x = x[train_x_size:train_x_size + 300, 100:]  # (batch_size, time-step) = (683, 120)
        test_y = y[train_x_size:train_x_size + 300, :]  # (batch_size, time-step) = (683, 1)

        pred = test_x.float().mean(dim=1).view(-1, 1)

        # mae = abs(test_y - pred).mean(dim=0).data.item()
        mse = abs(pow((test_y - pred), 2)).mean(dim=0).data.item()

        print(reg, cat, mse)
        print(reg, cat, mse, file=f)
