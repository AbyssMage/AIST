import os
import argparse
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from stgcn import STGCN
from utils import *
from torch.autograd import Variable
import time
import glob
import shutil
from sklearn.preprocessing import MinMaxScaler
import sys
import argparse as Ap
from mvgcn import *

# starts from here
seed = 0x6a09e667f3bcc908
np.random.normal(seed & 0xFFFFFFFF)
torch.manual_seed(seed & 0xFFFFFFFF)

argp = Ap.ArgumentParser()
argp.add_argument("--tr", default=7, type=int, help="Target region")
argp.add_argument("--tc", default=1, type=int, help="Target category")
d = argp.parse_args(sys.argv[1:])

target_region = 31  # 70, 42, 22, 24, 25, 26, 28, 29
target_cat = 1  # (starts from 0)
num_timesteps_input = 120
num_timesteps_output = 1
batch_size = 42

# laod Adjacency Matrix ( Works for MVGCN)
adj = gen_adj_matrix_STGCN(target_region)
train_x, train_d, train_w, train_y, test_x, test_d, test_w, test_y = load_data_regions_MVGCN(target_cat, target_region)
train_e, test_e = load_data_external_MVGCN(target_region)

print(train_x.shape, train_d.shape, train_w.shape, train_y.shape)
print(test_x.shape, test_d.shape, test_w.shape, test_y.shape)
print(train_e.shape, test_e.shape)

nhid = 22
n_recent = train_x.shape[3]
n_daily = train_d.shape[3]
n_weekly = train_w.shape[3]
n_ext = 12

model = MVGCN(nhid, n_recent, n_daily, n_weekly, n_ext)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
# criterion = nn.L1Loss()

loaded_data = torch.from_numpy(np.loadtxt("data/com_crime/r_" + str(target_region) + ".txt", dtype=np.int)).T
loaded_data = loaded_data[:, target_cat:target_cat+1]
tensor_ones = torch.from_numpy(np.ones((loaded_data.size(0), loaded_data.size(1)), dtype=np.int))
x, y = create_inout_sequences_GeoMAN(loaded_data, num_timesteps_input)
scale = MinMaxScaler(feature_range=(-1, 1))
x = torch.from_numpy(scale.fit_transform(x))
y = torch.from_numpy(scale.fit_transform(y))

train_batch = int(train_x.shape[0])
test_batch = int(test_x.shape[0])
print(train_batch)
print(test_batch)

epochs = 300  # 1000
training_losses = []
loss_values = []
validation_losses = []
validation_maes = []
patience = 100
best = epochs + 1
best_epoch = 0
t_total = time.time()
bad_counter = 0

for epoch in range(epochs):
    i = 0
    loss_values_batch = []
    for i in range(train_batch):  # 10
        t = time.time()

        x_r = Variable(train_x[i]).float()
        x_d = Variable(train_d[i]).float()
        x_w = Variable(train_w[i]).float()
        x_e = Variable(train_e[i]).float()
        y = Variable(train_y[i]).float()
        y = y[:, -1]

        # y = y.view(y.shape[0]).long()  # cross entropy, MAE loss works fine, does not work for MSE loss

        model.train()
        optimizer.zero_grad()
        output = model(adj, x_r, x_d, x_w, x_e)
        loss_train = criterion(output, y)
        loss_train.backward()
        optimizer.step()

        print('Epoch: {:04d}'.format(epoch*train_batch + i + 1),
              'loss_train: {:.4f}'.format(loss_train.data.item()),
              'time: {:.4f}s'.format(time.time() - t))

        loss_values.append(loss_train)
        torch.save(model.state_dict(), '{}.pkl'.format(epoch*train_batch + i + 1))
        if loss_values[-1] < best:
            best = loss_values[-1]
            best_epoch = epoch*train_batch + i + 1
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == patience:
            break

        files = glob.glob('*.pkl')
        for file in files:
            epoch_nb = int(file.split('.')[0])
            if epoch_nb < best_epoch:
                os.remove(file)

    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(file)

    if epoch*train_batch + i + 1 >= 800:
        break
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# best_epoch = 732
print('Loading {}th epoch'.format(best_epoch))
shutil.copy('{}.pkl'.format(best_epoch), 'E:/mvgcn/' + str(target_region) + '_' + str(target_cat) + '_mse.pkl')
model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))
# model.load_state_dict(torch.load("Heatmap/" + str(target_region) + "_" + str(target_cat) + ".pkl"))

"""print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    np.savetxt("wu.txt", model.state_dict()[param_tensor])"""

f = open('C:/Users/BD/PycharmProjects/AIST/mvgcn_eval_mse.txt','a')


def compute_test():
    loss = 0
    for i in range(test_batch):
        model.eval()

        test_x_r = Variable(test_x[i]).float()
        test_x_d = Variable(test_d[i]).float()
        test_x_w = Variable(test_w[i]).float()
        test_x_e = Variable(test_e[i]).float()
        y_test = Variable(test_y[i]).float()
        y_test = y_test[:, -1]
        # y_test = y_test.view(y_test.shape[0]).long()  # cross entropy

        # test_feat = region_daily_crime; test_feat_ext = external features
        output_test = model(adj, test_x_r, test_x_d, test_x_w, test_x_e)
        # print(attn_test.shape)

        """with open('E:/aist_datasets/base_prediction_trend/test_'+str(i)+'.pkl', 'wb') as f1:
            pickle.dump(output_test, f1)
        with open('E:/aist_datasets/base_attn_trend/test_' + str(i) + '.pkl', 'wb') as f2:
            pickle.dump(attn_test, f2)"""

        y_test = torch.from_numpy(scale.inverse_transform(y_test))
        output_test = torch.from_numpy(scale.inverse_transform(output_test.detach()))

        loss_test = criterion(output_test, y_test)

        for j in range(42):
            print(y_test[j, :].data.item(), output_test[j, :].data.item())

        loss += loss_test.data.item()
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.data.item()))

    print(loss/i)
    print(target_region, " ", target_cat, " ", loss/i, file=f)


compute_test()

