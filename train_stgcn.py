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
# starts from here
seed = 0x6a09e667f3bcc908
np.random.normal(seed & 0xFFFFFFFF)
torch.manual_seed(seed & 0xFFFFFFFF)

argp = Ap.ArgumentParser()
argp.add_argument("--tr", default=7, type=int, help="Target region")
argp.add_argument("--tc", default=1, type=int, help="Target category")
d = argp.parse_args(sys.argv[1:])

target_region = 43  # 65 66 68 70 7 42 43 24 28
target_cat = 3  # (starts from 0)
num_timesteps_input = 120
num_timesteps_output = 1
batch_size = 42

# laod Adjacency Matrix
A = gen_adj_matrix_STGCN(target_region)
A_wave = get_normalized_adj_STGCN(A)
A_wave = torch.from_numpy(A_wave)
A_wave = Variable(A_wave).float()
# A = np.loadtxt("data/com_adj_matrix.txt", dtype=np.int)
# print(A)
# A = A[:5, : 5]

# load_data - crime features
# train_x, train_y, test_x, test_y = load_data_all_STGCN(target_cat)
train_x, train_y, test_x, test_y = load_data_regions_STGCN(target_cat, target_region)
print(train_x.dtype, train_y.dtype, A_wave.dtype)

# train_x, train_y = train_x.type(torch.DoubleTensor), train_y.type(torch.DoubleTensor)

training_input, training_target = train_x, train_y
test_input, test_target = test_x, test_y

# print(train_x.shape, train_y.shape)
# print(test_x.shape, test_y.shape)

lr = 0.003
net = STGCN(A_wave.shape[0], training_input.shape[3], num_timesteps_input, num_timesteps_output)
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
# loss_criterion = nn.MSELoss()
loss_criterion = nn.L1Loss()

loaded_data = torch.from_numpy(np.loadtxt("data/com_crime/r_" + str(target_region) + ".txt", dtype=np.int)).T
loaded_data = loaded_data[:, target_cat:target_cat+1]
tensor_ones = torch.from_numpy(np.ones((loaded_data.size(0), loaded_data.size(1)), dtype=np.int))
x, y = create_inout_sequences_GeoMAN(loaded_data, num_timesteps_input)
scale = MinMaxScaler(feature_range=(-1, 1))
x = torch.from_numpy(scale.fit_transform(x))
y = torch.from_numpy(scale.fit_transform(y))

train_batch = int(train_x.shape[0] / batch_size)
test_batch = int(test_x.shape[0] / batch_size)
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

for epoch in range(epochs):
    train_x = Variable(train_x).float()
    train_y = Variable(train_y).float()

    permutation = torch.randperm(train_x.shape[0])
    epoch_training_losses = []
    cnt = 0
    for i in range(0, train_x.shape[0], batch_size):
        t = time.time()

        net.train()
        optimizer.zero_grad()

        indices = permutation[i:i + batch_size]
        X_batch, y_batch = train_x[indices], train_y[indices]
        
        out = net(A_wave, X_batch)
        out = out[:, -1, :]
        y_batch = y_batch[:, -1, :]

        # for j in range(42):
            # print(y_batch[j, :].data.item(), out[j, :].data.item())

        loss = loss_criterion(out, y_batch)
        loss.backward()
        optimizer.step()
        epoch_training_losses.append(loss.detach().cpu().numpy())

        print('Epoch: {:04d}'.format(epoch * train_batch + cnt + 1),
              'loss_train: {:.4f}'.format(loss.data.item()),
              'time: {:.4f}s'.format(time.time() - t))

        loss_values.append(loss)

        torch.save(net.state_dict(), '{}.pkl'.format(epoch * train_batch + cnt + 1))

        if loss_values[-1] < best:
            best = loss_values[-1]
            best_epoch = epoch * train_batch + cnt + 1
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

        cnt = cnt + 1
    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(file)

    if epoch * train_batch + cnt + 1 >= 800:
        break
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# best_epoch = 402
print('Loading {}th epoch'.format(best_epoch))
shutil.copy('{}.pkl'.format(best_epoch), 'F:/STGCN/' + str(target_region) + '_' + str(target_cat) + '_mae.pkl')
net.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))
# model.load_state_dict(torch.load("Heatmap/" + str(target_region) + "_" + str(target_cat) + ".pkl"))

f = open('C:/Users/BD/PycharmProjects/AIST/stgcn_eval_mae.txt', 'a')

test_x = test_x.view(int(test_x.shape[0] / batch_size), batch_size, A.shape[0], num_timesteps_input, 1)
test_y = test_y.view(int(test_y.shape[0] / batch_size), batch_size, A.shape[0], 1)


def compute_test():
    loss = 0
    for i in range(test_batch):
        net.eval()

        test_X_batch = Variable(test_x[i]).float()
        test_y_batch = Variable(test_y[i]).float()

        out_test = net(A_wave, test_X_batch)
        out_test = out_test[:, -1, :].view(batch_size, -1)
        test_y_batch = test_y_batch[:, -1, :]
        test_y_batch = test_y_batch.view(-1, 1)

        """with open('E:/aist_datasets/base_prediction_trend/test_'+str(i)+'.pkl', 'wb') as f1:
            pickle.dump(output_test, f1)
        with open('E:/aist_datasets/base_attn_trend/test_' + str(i) + '.pkl', 'wb') as f2:
            pickle.dump(attn_test, f2)"""

        test_y_batch = torch.from_numpy(scale.inverse_transform(test_y_batch))
        out_test = torch.from_numpy(scale.inverse_transform(out_test.detach()))

        loss_test = loss_criterion(out_test, test_y_batch)

        for j in range(42):
            print(test_y_batch[j, :].data.item(), out_test[j, :].data.item())

        loss += loss_test.data.item()
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.data.item()))

    print(loss/i)
    print(target_region, " ", target_cat, " ", loss / i)
    print(target_region, " ", target_cat, " ", loss/i, file=f)


compute_test()
