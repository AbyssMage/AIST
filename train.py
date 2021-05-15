import time
import torch.optim as optim
import glob
import os
from model import *
from layers import *
from sklearn.preprocessing import MinMaxScaler
import sys
import argparse as Ap

seed = 0x6a09e667f3bcc908
np.random.normal(seed & 0xFFFFFFFF)
torch.manual_seed(seed & 0xFFFFFFFF)

argp = Ap.ArgumentParser()
argp.add_argument("--tr", default=7, type=int, help="Target region")
argp.add_argument("--tc", default=1, type=int, help="Target category")
argp.add_argument("--ts", default=20, type=int, help="Number of time-steps")
argp.add_argument("--in_hgat", default=1, type=int, help="input dimension of hgat layers")
argp.add_argument("--in_fgat", default=12, type=int, help="input dimension of fgat layers:num of features")
argp.add_argument("--out_gat", default=8, type=int, help="output dimension of both hgat and fgat layers")
argp.add_argument("--att_dot",default=40,type=int, help="(dot-product)attention dimension of fgat")
argp.add_argument("--nhid_rnn",default=40,type=int, help="hidden dimension of rnn")
argp.add_argument("--nlayer_rnn",default=1,type=int, help="number of rnn layers")
argp.add_argument("--att_rnn",default=30,type=int, help="(location)attention dimension of temporal module")

d = argp.parse_args(sys.argv[1:])

target_region = d.tr
target_cat = d.tc
ts = d.ts
in_hgat = d.in_hgat
in_fgat = d.in_fgat
out_gat = d.out_gat
att_dot = d.att_dot
nhid_rnn = d.nhid_rnn
nlayer_rnn = d.nlayer_rnn
att_rnn = d.att_rnn
batch_size = 42

gen_gat_adj_file(target_region)   # generate the adj_matrix file for GAT layers
loaded_data = torch.from_numpy(np.loadtxt("data/com_crime/r_" + str(target_region) + ".txt", dtype=np.int)).T
loaded_data = loaded_data[:, target_cat:target_cat+1]
tensor_ones = torch.from_numpy(np.ones((loaded_data.size(0), loaded_data.size(1)), dtype=np.int))
x, y, x_daily, x_weekly = create_inout_sequences(loaded_data)


scale = MinMaxScaler(feature_range=(-1, 1))
x = torch.from_numpy(scale.fit_transform(x))
x_daily = torch.from_numpy(scale.fit_transform(x_daily))
x_weekly = torch.from_numpy(scale.fit_transform(x_weekly))
y = torch.from_numpy(scale.fit_transform(y))

# load data
train_x, train_x_daily, train_x_weekly, train_y, test_x, test_x_daily, test_x_weekly, test_y = load_self_crime(x, x_daily, x_weekly, y)
train_nei, test_nei = load_nei_crime(target_cat, target_region)
train_ext, test_ext = load_all_ext(target_region)
train_side, test_side = load_all_parent_crime(target_cat, target_region)

# Model and optimizer
model = AIST(in_hgat, in_fgat, out_gat, att_dot, nhid_rnn, nlayer_rnn, att_rnn, ts, target_region, target_cat)
n = sum(p.numel() for p in model.parameters() if p.requires_grad)

lr = 0.001 # initial learning rate 0.004
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.L1Loss()

epochs = 300
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

        x_crime = Variable(train_x[i]).float()
        x_crime_daily = Variable(train_x_daily[i]).float()
        x_crime_weekly = Variable(train_x_weekly[i]).float()
        y = Variable(train_y[i]).float()

        model.train()
        optimizer.zero_grad()
        output, attn = model(x_crime, x_crime_daily, x_crime_weekly, train_nei[i], train_ext[i], train_side[i])
        y = y.view(-1, 1)

        loss_train = criterion(output, y)
        loss_train.backward()
        optimizer.step()

        """print('Epoch: {:04d}'.format(epoch*train_batch + i + 1),
              'loss_train: {:.4f}'.format(loss_train.data.item()),
              'time: {:.4f}s'.format(time.time() - t))"""

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

# best_epoch = 666
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))


def compute_test():
    loss = 0
    for i in range(test_batch):
        model.eval()

        x_crime_test = Variable(test_x[i]).float()
        x_crime_daily_test = Variable(test_x_daily[i]).float()
        x_crime_weekly_test = Variable(test_x_weekly[i]).float()
        y_test = Variable(test_y[i]).float()

        output_test, attn_test = model(x_crime_test, x_crime_daily_test, x_crime_weekly_test, test_nei[i], test_ext[i], test_side[i])
        y_test = y_test.view(-1, 1)

        y_test = torch.from_numpy(scale.inverse_transform(y_test))
        output_test = torch.from_numpy(scale.inverse_transform(output_test.detach()))

        loss_test = criterion(output_test, y_test)

        """for j in range(42):
            print(y_test[j, :].data.item(), output_test[j, :].data.item())"""

        loss += loss_test.data.item()
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.data.item()))

    print(target_region, " ", target_cat, " ", loss/i)


compute_test()

