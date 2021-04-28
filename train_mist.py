import numpy as np
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import glob
import os
from utils import *
from model import *
from layers import *
from sklearn.preprocessing import MinMaxScaler
import shutil
seed = 0x6a09e667f3bcc908
np.random.normal(seed & 0xFFFFFFFF)
torch.manual_seed(seed & 0xFFFFFFFF)

# load crime data
time_step = 120
batch_size = 42
target_region = 7
target_cat = 0

loaded_data = torch.from_numpy(np.loadtxt("data/com_crime/r_" + str(target_region) + ".txt", dtype=np.int)).T
loaded_data = loaded_data[:, target_cat:target_cat+1]
tensor_ones = torch.from_numpy(np.ones((loaded_data.size(0), loaded_data.size(1)), dtype=np.int))
# loaded_data = torch.where(loaded_data > 1, tensor_ones, loaded_data)
x, y, x_daily, x_weekly = create_inout_sequences(loaded_data, time_step)

scale = MinMaxScaler(feature_range=(-1, 1))
x = torch.from_numpy(scale.fit_transform(x))
x_daily = torch.from_numpy(scale.fit_transform(x_daily))
x_weekly = torch.from_numpy(scale.fit_transform(x_weekly))
y = torch.from_numpy(scale.fit_transform(y))

# Divide into train_test data
train_x_size = int(x.shape[0] * .67)  # batch_size
train_x = x[: train_x_size, :]  # (batch_size, time-step) = (1386, 120)
train_x_daily = x_daily[: train_x_size, :]
train_x_weekly = x_weekly[: train_x_size, :]
train_y = y[: train_x_size, :]  # (batch_size, time-step) = (1386, 1)

test_x = x[train_x_size:, :]  # (batch_size, time-step) = (683, 120)
test_x_daily = x_daily[train_x_size:, :]
test_x_weekly = x_weekly[train_x_size:, :]
test_x = test_x[:test_x.shape[0] - 11, :]  # (batch_size, time-step) = (672, 120) -- to make it consistent with the batch size
test_x_daily = test_x_daily[:test_x_daily.shape[0] - 11, :]
test_x_weekly = test_x_weekly[:test_x_weekly.shape[0] - 11, :]
test_y = y[train_x_size:, :]  # (batch_size, time-step) = (683, 1)
test_y = test_y[:test_y.shape[0] - 11, :]

# Divide it into batches -----> (Num of Batches, batch size, time-step features)
train_x = train_x.view(int(train_x.shape[0] / batch_size), batch_size, time_step)
train_x_daily = train_x_daily.view(int(train_x_daily.shape[0] / batch_size), batch_size, train_x_daily.shape[1])
train_x_weekly = train_x_weekly.view(int(train_x_weekly.shape[0] / batch_size), batch_size, train_x_weekly.shape[1])
train_y = train_y.view(int(train_y.shape[0] / batch_size), batch_size, 1)

test_x = test_x.view(int(test_x.shape[0] / batch_size), batch_size, time_step)
test_x_daily = test_x_daily.view(int(test_x_daily.shape[0] / batch_size), batch_size, test_x_daily.shape[1])
test_x_weekly = test_x_weekly.view(int(test_x_weekly.shape[0] / batch_size), batch_size, test_x_weekly.shape[1])
test_y = test_y.view(int(test_y.shape[0] / batch_size), batch_size, 1)

# load data for MiST
train_mist, test_mist = load_crime_data_regions_MiST2(target_region, target_cat)
# feat_ext = torch.stack(train_mist[0], 0)
# print(feat_ext.shape)
# feat_ext = torch.transpose(feat_ext, 2, 1)
# feat_ext = torch.transpose(feat_ext, 1, 0)

# x = torch.stack(train_mist[0], dim=1)

# parameters for MiST
nfeature = 1  # num of feature per time-step
nhid = 32  # num of features in hidden dimensions of rnn MiST paper = 32

model = MiST(nfeature, nhid)
print(model)
n = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(n)

lr = 0.001  # MiST paper = 0.001
weight_decay = 5e-4  # Weight decay = (L2 loss on parameters)
optimizer = optim.Adam(model.parameters(), lr=lr)
# criterion = nn.CrossEntropyLoss()
# criterion = nn.MSELoss()
criterion = nn.L1Loss()

epochs = 500
best = epochs + 1
best_epoch = 0
t_total = time.time()
loss_values = []
bad_counter = 0
patience = 100

train_batch = train_x.shape[0]
test_batch = test_x.shape[0]
print(train_batch, test_batch)

for epoch in range(epochs):
    i = 0
    loss_values_batch = []
    for i in range(train_batch):
        t = time.time()
        # x_crime = Variable(train_x[i]).float()
        y = Variable(train_y[i]).float()
        # y = y.view(y.shape[0]).long()  # cross entropy, MAE loss does not work for MSE Loss

        model.train()
        optimizer.zero_grad()
        output = model(train_mist[i])  # .squeeze(1)
        # print(output.shape, y.shape)

        loss_train = criterion(output, y)
        loss_train.backward()
        optimizer.step()

        print('Epoch: {:04d}'.format(epoch * train_batch + i + 1),
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
                # file.close()
                os.remove(file)

    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb > best_epoch:
            # file.close()
            os.remove(file)

    if epoch*train_batch + i + 1 >= 700:
        break

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# best_epoch = 100
print('Loading {}th epoch'.format(best_epoch))
# shutil.copy('{}.pkl'.format(best_epoch), 'E:/MiST_PKL/' + str(target_region) + '_' + str(target_cat) + '_MAE.pkl')
model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))


# x_poi_test = Variable(torch.from_numpy(poi).repeat(test_x.shape[0], test_x.shape[1], 1)).float()
def compute_test():
    loss = 0
    for i in range(test_batch):
        model.eval()
        # x_crime_test = test_x.view(test_x.shape[0], test_x.shape[1], 1)
        # x_crime_test = Variable(x_crime_test).float()
        # x_crime_test = torch.cat([x_crime_test, x_poi_test], dim=2)
        # x_crime_test = Variable(test_x[i]).float()
        y_test = Variable(test_y[i]).float()

        # y_test = y_test.view(y_test.shape[0]).long()  # cross entropy
        output_test = model(test_mist[i])

        y_test = torch.from_numpy(scale.inverse_transform(y_test))
        output_test = torch.from_numpy(scale.inverse_transform(output_test.detach()))

        loss_test = criterion(output_test, y_test)
        loss += loss_test.data.item()
        for j in range(42):
            print(y_test[j, :].data.item(), output_test[j, :].data.item())
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.data.item()))
        """print("Test set results:",
              "loss= {:.4f}".format(loss_test.data.item()),
              'macro_f1: {:.4f}'.format(macro_f1_test),
              'micro_f1: {:.4f}'.format(micro_f1_test),
              "accuracy= {:.4f}".format(acc_test.data.item()))"""

    """print("accuracy = {:.4f}".format((acc/i).data.item()),
          'macro_f1: {:.4f}'.format(mac_f1/i),
          'micro_f1: {:.4f}'.format(mic_f1/i))"""
    print(loss/i)


compute_test()

