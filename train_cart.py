from sklearn import tree
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import math
from io import StringIO
import time
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from utils import *
import torch
# c1_mae = {0, 5, 6, 7, 71, 21, 22, 23, 24, 27, 31}
# c2_mae = {66, 68, 70, 7, 71, 42, 43, 48, 22, 23, 24}
# c3_mae = {66, 68, 70, 7, 71, 42, 43, 22, 24, 28}

time_step = 120
n_samples = 1386
# 66, 68, 70, 7, 71, 42, 43, 22, 24, 28
target_region = [0, 5, 6, 7, 71, 21, 22, 23, 24, 27, 31]  # (starts from 0)
target_cat = 1  # (starts from 0)
for tr in target_region:
    loaded_data = torch.from_numpy(np.loadtxt("data/com_crime/r_" + str(tr) + ".txt", dtype=np.int)).T
    loaded_data = loaded_data[:, target_cat:target_cat+1]
    x, y, x_daily, x_weekly = create_inout_sequences(loaded_data, time_step)
    x = x[:, 100:]

    x = x.numpy()
    y = y.numpy()

    train_x = x[:n_samples, :]
    train_y = y[:n_samples, :]
    test_x = x[n_samples:, :]
    test_y = y[n_samples:, :]

    clf = tree.DecisionTreeRegressor(criterion="mae", max_depth=None)
    clf = clf.fit(train_x, train_y)
    y_hat = clf.predict(test_x)
    loss_mae = mean_absolute_error(test_y, y_hat)

    clf = tree.DecisionTreeRegressor(criterion="mse", max_depth=None)
    clf = clf.fit(train_x, train_y)
    y_hat = clf.predict(test_x)
    loss_mse = mean_squared_error(test_y, y_hat)

    f = open('C:/Users/Yeasir Rayhan Prince/PycharmProjects/AIST/cart.txt', 'a')
    print(tr, target_cat, loss_mae, loss_mse, file=f)
    print(tr, target_cat, loss_mae, loss_mse)









