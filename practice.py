import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import shutil
import utils
from utils import *
np.set_printoptions(threshold=np.inf)

target_region = [22]
target_cat = 2
time_step = 120
month = 6
for i in target_region:
    loaded_data = torch.from_numpy(np.loadtxt("data/com_crime/r_" + str(i) + ".txt", dtype=np.int)).T
    loaded_data = loaded_data[:, target_cat:target_cat+1]
    tensor_ones = torch.from_numpy(np.ones((loaded_data.size(0), loaded_data.size(1)), dtype=np.int))
    # loaded_data = torch.where(loaded_data > 1, tensor_ones, loaded_data)  # Needed for classification Problem
    x, y, x_daily, x_weekly = create_inout_sequences(loaded_data, time_step)

    single = int(x.shape[0]/12)
    for j in range(12):
        tem1 = x[single * j : single* (j+1), :]
        tem1 = torch.sum(tem1)
        print(tem1)

exit()
cat = [1, 2, 3, 7]
reg = [7]
x = np.loadtxt("data/com_crime/r_" + str(0) + ".txt", dtype=np.int)
print(x.shape)

exit()

x = np.loadtxt("data/com_crime/r_" + str(0) + ".txt", dtype=np.int)
print(x.shape)

arr = np.ones((77, 2190), dtype=np.int)
for i in range(77):
    x = np.loadtxt("data/com_crime/r_" + str(i) + ".txt", dtype=np.int)
    arr[i] = x[7]
arr = np.sum(arr, axis=0)
arr_daily = arr.reshape((365, 6))
print(arr_daily)
arr_ts = np.sum(arr_daily, axis=0)
print(arr_ts)

arr_w = np.sum(arr_daily, axis=1)
arr_w = arr_w[:364]
arr_w = arr_w.reshape((52, 7))
print(arr_w)
arr_w = np.sum(arr_w, axis=0)
print(arr_w)
exit(0)


attn = np.zeros((10, 20))
inv_l = 1. / (20 - 2)
attn += inv_l[:, None]
attn = torch.Tensor(attn)
print(attn)
exit(0)

attn = np.zeros((20, 5))
inv_l = 1. / (self.lengths.cpu().data.numpy() - 2)
attn += inv_l[:, None]
attn = torch.Tensor(attn).to(device)
attn.masked_fill_(self.masks, 0)

tr = 72
x = utils.gen_neighbor_index_zero_with_target(tr)
for j in x:
    c = np.loadtxt("data/com_crime/r_" + str(j) + ".txt", dtype=np.int)
    print(j+1, ":", c[1].sum(), c[2].sum(), c[3].sum(), c[7].sum())
exit(0)
"""bucket = []
cat = [0, 7, 3]
for i in range(77):
    x = np.loadtxt("data/com_crime/r_" + str(i) + ".txt", dtype=np.int)
    for j in cat:
        if x[j].sum() in range(500, 5000):
            bucket.append(i)
            break
x = set(bucket)
print(x)
exit(0)
exit(0)"""
com = []
for i in range(77):
    x = np.loadtxt("data/com_crime/r_" + str(i) + ".txt", dtype=np.int)
    if x[4].sum() + x[5].sum() in range(1000, 10000):
        print(i, x[4].sum() + x[5].sum())
        com.append(i)
com.sort()
print(com)
exit(0)


best_epoch = 830
shutil.copy('{}.pkl'.format(best_epoch), 'E:/aist_datasets/' + str(8)+'.pkl')
exit(0)

"""a = Variable(torch.tensor([0.4, 0.2, 0.4]))
b = Variable(torch.tensor([0, 0, 1]))
criterion = nn.KLDivLoss()
loss = criterion(a.log(), b)
print(loss)
exit(0)
B = 42
N = 5
F = 12
FP = 8
dq = 8
dk = 8
a = torch.zeros((B, N, N, F))
b = torch.zeros((B, N, F, FP))
c = torch.matmul(a, b)
c = c.sum(dim=2)
print(c.shape)"""
exit(0)

# a = a.repeat(1, 1, 1, F).view(B, N, N, F, dq)
print(a.shape, b.shape)
c = torch.matmul(b, a)
print(c.shape)
# torch.matmul()

exit(0)

# Generate Key --> (N, N, num_feat, 2*F')
# ext_input = ext_input.long()
"""ext_input = ext_input.view(ext_input.shape[0], nfeature, 1)
# hf = self.embed(ext_input)  # (N, nfeat, F')
hf = self.WF(ext_input)
hf = hf.view(N, -1)  # (N, nfeat*F')
a_input_f = torch.cat([hf.repeat(1, N).view(N * N, -1), hf.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)  # (N, N, 2*nfeat*F')
k = a_input_f.view(N*N*nfeature, -1)  # (N*N*feat, 2F')
k = self.WK(k)  # (N*N*feat, dk)
k = k.view(N*N, nfeature, -1)  # (N*N, feat, dk)
k = torch.transpose(k, 2, 1)
# print(q.shape, k.shape)

dot_attention = torch.bmm(q, k).squeeze(1)  # (N*N, nfeat)
zero_vec = -9e15 * torch.ones_like(dot_attention)  # shape = (N*N, nfeat)
dot_attention = torch.where(dot_attention > 0, dot_attention, zero_vec)  # (N*N, nfeat)
dot_attention = F.softmax(dot_attention, dim=1)  # shape = (N*N, nfeat)
dot_attention = dot_attention.view(N, N, -1)
# print(dot_attention[-1, 205:209, :].detach().numpy())
dot_attention = dot_attention.view(N, N * nfeature)
# np.savetxt("AIST_DOT_ATTENTION.txt", dot_attention.view(N * N, -1).detach().numpy(), fmt="%.4f")

# Generate Value
hf = hf.view(N, nfeature, -1)  # (N, nfeat, F')
v = self.WV(hf)  # (N, nfeat, dv)
v = v.view(N*nfeature, -1)

# final ext_rep(new)
reg_attention = attention.view(N*N, 1)
# print(reg_attention.shape)
reg_attention = reg_attention.repeat(1, nfeature)
reg_attention = reg_attention.view(N, N, -1)
dot_attention = dot_attention.view(N, N, -1)

total_attention = dot_attention*reg_attention
total_attention = F.softmax(total_attention, dim=1)  # shape = (N*N, nfeat)
total_attention = total_attention.view(N, N * nfeature)
ext_rep = torch.matmul(total_attention, v)  # (N, N, dv)"""


a = torch.ones((42, 3))
a = torch.softmax(a, dim=1)
print(a)
print(a.shape)
exit(0)
W = torch.zeros(size=(1, 8))
a = torch.zeros(size=(16, 1))
input = torch.zeros(size=(42, 5, 1))
h = torch.matmul(input, W)
N = h.size()[1]  # N = Number of Nodes
print(h.repeat(1, 1, N).view(42, N * N, -1).shape)
print(h.repeat(1, N, 1).shape)
a_input = torch.cat([h.repeat(1, 1, N).view(42, N * N, -1), h.repeat(1, N, 1)], dim=2).view(42, N, -1, 2 * 8)
e = torch.matmul(a_input, a)
print(e.shape)
exit(0)

# h_prime = torch.matmul(attention, h)  # shape = (N, F'1)

from sklearn.preprocessing import MinMaxScaler
import numpy as np
x = np.ndarray([1, 10, 4, 0, 5, 9])
scale = MinMaxScaler(feature_range=(0, 1))
x = scale.fit_transform(x.reshape(-1, 1))
print(x)
exit(0)
import pandas as pd
from sodapy import Socrata
import torch
import numpy as np
import torch.nn
N = 3
nfeat = 6
E = 4

f = torch.rand((N, nfeat, 1))
l = torch.nn.Linear(1, E)
f = l(f)
print(f.shape)
exit(0)

N = 3
nfeat = 5
dv = 2
att = torch.rand((N, N, nfeat))
x = torch.rand((N, nfeat, dv))
print(att.shape, x.shape)
ans = torch.matmul(att, x)
print(ans.shape)
exit(0)


dot = torch.tensor([[0.2, 0.4, 0.3, 0.1], [0.5, 0.3, 0.1, 0.2]])
att = torch.tensor([[0.8], [0.2]])
att = att.repeat(1, 4).view(2, 4)
print(att)
print(dot.shape, att.shape)
ans = dot*att
print(ans.shape)
print(ans)
exit(0)

N = 3
F = 4
feat_num = 5
A = 10
h = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
h = torch.cat((h.repeat(1, N).view(N*N, -1), h.repeat(N, 1)), dim=1)
h = h.view(N*N, 2*F).float()
h = torch.nn.Linear(2*F, A)(h)
h = h.unsqueeze(1)
print(h.shape)
h_f = torch.from_numpy(np.zeros((N, feat_num), dtype=int)).long()
h_f = torch.nn.Embedding(feat_num, F)(h_f)
h_f = h_f.view(N, -1)
h_f = torch.cat((h_f.repeat(1, N).view(N*N, -1), h_f.repeat(N, 1)), dim=1)
h_f = h_f.view(N*N*feat_num, -1)
h_f = torch.nn.Linear(2*F, A)(h_f)
h_f = h_f.view(N*N, feat_num, -1)
print(h_f.shape)

ans = torch.bmm(h, h_f)
# print(ans.shape)
exit(0)


B = 3
V = 4
C = 5
E = 8
T = 2
N = 3
F = 3
FP = 8
Voc = 10
E = 6
A = 3

embedding = torch.nn.Embedding(Voc, E)
x = torch.from_numpy(np.zeros((B*V, C), dtype=int)).long()
W = torch.nn.Linear(E, A, bias=False)

word_embed = embedding(x)
Q = W(word_embed)
K = W(word_embed)
K = torch.transpose(K, 2, 1)
Ans = torch.bmm(Q, K)
print(Q.shape, K.shape, Ans.shape)


exit(0)


label = torch.tensor([6, 23, 27, 31, 7])
label = label.repeat(T*B, 1)  # (T*B, N)
label = label.view(label.shape[0] * label.shape[1], 1)  # (T*B*N, 1)
print(label)
label = label.view(T, B * 5, 1)  # (T*B*N, 1)
print(label)

h = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
r1 = h.repeat(1, N).view(N*N, -1)
r2 = h.repeat(N, 1)
print(r1, r2)
torch.nn.Embedding
"""
torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * FP)
w = torch.from_numpy(np.zeros((F, FP), dtype=np.int))
h = torch.mm(inp, w)
N = h.size()[0]  # N = Number of Nodes
a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * FP)
a = torch.from_numpy(np.zeros((2*FP, 1), dtype=np.int))
e = torch.matmul(a_input, a).squeeze(2)
"""
"""
inp = torch.from_numpy(np.zeros((B * N, F), dtype=np.int))
w = torch.from_numpy(np.zeros((F, FP), dtype=np.int))
h = torch.mm(inp, w)
N = h.size()[0]  # N = Number of Nodes
a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * FP)
a = torch.from_numpy(np.zeros((2*FP, 1), dtype=np.int))
e = torch.matmul(a_input, a).squeeze(2)
print(e.shape)
"""
"""
N = h.size()[0]  # N = Number of Nodes

# h.repeat(1, N).view(N * N, -1) = (NxN, F'), h.repeat(N, 1) = (NxN, F')
# cat = (NxN, 2F')
# a_input = (N, N, 2F')
a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
# torch.matmul(a_input, self.a).squeeze(2) = ((N, N, 1) -----> (N, N))
e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

zero_vec = -9e15 * torch.ones_like(e)  # shape = (N, N)
attention = torch.where(adj > 0, e, zero_vec)  # shape = (N, N)
# print(attention)
attention = F.softmax(attention, dim=1)  # shape = (N, N)
attention = F.dropout(attention, self.dropout, training=self.training)  # shape = (N, N)

h_prime = torch.matmul(attention, h)  # shape = (N, F')

if self.concat:
            return F.elu(h_prime)
else:
            return h_prime
"""
"""
# Unauthenticated client only works with public data sets. Note 'None'
# in place of application token, and no username or password:
# client = Socrata("data.cityofchicago.org", None)

# Example authenticated client (needed for non-public datasets):
client = Socrata("data.cityofchicago.org",
                  app_token="Zmx9MWD39q6ACpbcrChDNaG3O",
                  username="yeasirrayhan.prince@gmail.com",
                  password="buet241@PRINCE")
client.timeout = 3600
# First 2000 results, returned as JSON from API / converted to Python list of
# dictionaries by sodapy.
results = client.get("h4cq-z3dy", limit=1000000)
# Convert to pandas DataFrame
results_df = pd.DataFrame.from_records(results)
print(results_df)
results_df.to_csv("F:/crime_dataset/Actual Chicago Data/N.csv")
print(results_df.shape)
"""