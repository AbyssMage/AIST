import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import shutil
import utils
"""model = np.loadtxt("model_results_mse.txt", dtype=np.float)
model = model.T
for i in range(model.shape[0]):
    print(np.argsort(model[i]))
exit(0)"""

"""geoman_mse = [1.9930476982177245, 0.43849810604866696, 1.05808100321516, 0.5594614730193689]
geoman_mae = [0.9091687032084745, 0.3876142363721158, 0.7226571661162943, 0.3449679689002888]


arnn_mae = [1.0419, 0.4096, 0.7377, 0.4128]
arnn_mse = [2.5443, 0.4427, 1.0665, 0.6380]

stgcn_mae = np.array([1.0416, 0.5130, 1.0869, 0.3886])
stgcn_mse =	np.array([1.8121, 0.4860, 1.0595, 0.6342])

mist_mae = np.array([1.0241, 0.3727, 0.7365, 0.3701])
mist_mse =	np.array([2.5153, 0.4836, 1.0345, 0.6495])

dc_mae = np.array([1.0022, 0.3727, 0.7271, 0.3702])
dc_mse = np.array([2.6279, 0.4751, 1.0567, 0.6394])

print("Att-RNN", np.sum(arnn_mae, axis=0)/4)
print("DeepCrime", np.sum(dc_mae, axis=0)/4)
print("MiST", np.sum(mist_mae, axis=0)/4)
print("STGCN", np.sum(stgcn_mae, axis=0)/4)
print("GeoMAN", np.sum(geoman_mae, axis=0)/4)
print("\n")
print("Att-RNN", np.sum(arnn_mse, axis=0)/4)
print("DeepCrime", np.sum(dc_mse, axis=0)/4)
print("MiST", np.sum(mist_mse, axis=0)/4)
print("STGCN", np.sum(stgcn_mse, axis=0)/4)
print("GeoMAN", np.sum(geoman_mse, axis=0)/4)
exit()"""

# Cal Parts of the Model
"""cat = [1, 2, 3, 7]
model = np.loadtxt("sab_r_mae.txt", dtype=np.float)
for j in cat:
    raw = []
    for k in range(model.shape[0]):
        if model[k][1] == j:
            raw.append(model[k][2])
    # print(raw)
    print(len(raw))
    print(j, sum(raw)/len(raw))
exit(0)"""

# Cal Param
"""cat = [3, 7]
nhid = [36, 44, 48]
model = np.loadtxt("aist_param_Q_mae.txt", dtype=np.float)
for i in nhid:
    for j in cat:
        raw = []
        for k in range(model.shape[0]):
            if model[k][0] == i and model[k][2] == j:
                raw.append(model[k][3])
        # print(raw)
        print(len(raw))
        print(i, j, sum(raw)/len(raw))
exit(0)"""

# Train / Test Ratio
"""ratio = [0.5, 0.5833, 0.75, 0.8333]
cat = 7
model = np.loadtxt("aist_train_test_mae.txt", dtype=np.float)
for i in ratio:
    raw = []
    for j in range(model.shape[0]):
        if model[j][0] == i and model[j][2] == cat:
            raw.append(model[j][3])
    # print(raw)
    print(len(raw))
    if len(raw) != 0:
        print(i, sum(raw)/len(raw))
exit(0)"""

# cal eval_model
cat = [2]
model = np.loadtxt("geoman_eval_mse.txt", dtype=np.float)
model1 = np.loadtxt("aist_eval_module_mse.txt", dtype=np.float)
for i in cat:
    raw = []
    com = []
    for j in range(model.shape[0]):
        if model[j][1] == i:
            raw.append(model[j][2])
    print(raw)
    print(len(raw))
    print(i, sum(raw)/len(raw))

for i in cat:
    raw = []
    com = []
    for j in range(model1.shape[0]):
        if model1[j][1] == i:
            raw.append(model1[j][2])
    print(raw)
    print(len(raw))
    print(i, sum(raw)/len(raw))
exit()

a = [0.8746520970285498, 0.36148352331171435,0.6910386574634957, 0.3399464720120978]
mae_aist = sum(a)/len(a)
b = [1.0240649929688996, 0.3727438937050182, 0.7365264269272014, 0.3701086623671036]
mae_mist = sum(b)/len(b)
c = [1.0022495290075721, 0.37273496678273743, 0.7270856707141039, 0.3701938268364895]
mae_dc = sum(c)/len(c)
geoman_mae = [0.9091687032084745, 0.3876142363721158, 0.7226571661162943, 0.3449679689002888]
mae_geoman = sum(geoman_mae) / len(geoman_mae)

# print(mae_aist, mae_mist, mae_dc, mae_geoman)
# print(mae_aist/mae_dc)

decrease = (mae_geoman - mae_aist) / mae_geoman
print(decrease)

geoman_mse = [1.9930476982177245, 0.43849810604866696, 1.05808100321516, 0.5594614730193689]
mse_geoman = sum(geoman_mse)/len(geoman_mse)

d = [1.6985768375896406, 0.48369775578765306, 0.956839547992963, 0.5609723588785754]
mse_aist = sum(d)/len(d)
e = [2.515321472539998, 0.48362364201776, 1.0344653130032315, 0.6495051331152005]
mse_mist = sum(e)/len(e)
f = [2.6279659749181192, 0.4750698739945943, 1.0567111700418839, 0.639412143616841]
mse_dc = sum(f)/len(f)
# print(mse_aist, mse_mist, mse_dc, mse_geoman)
# print(mse_aist/mse_mist)


decrease = (mse_geoman - mse_aist) / mse_mist
print(decrease)
exit(0)
# cal eval_model
cat = [1, 2, 3, 7]
aist = np.loadtxt("gat+f_eval_mae.txt", dtype=np.float)
for i in cat:
    raw = []
    com = []
    for j in range(aist.shape[0]):
        if aist[j][1] == i:
            raw.append(aist[j][2])
    # print(raw)
    print(len(raw))
    print(i, sum(raw)/len(raw))
exit()


# cal eval_param --- H
param = [0, 1, 2, 3]
aist = np.loadtxt("aist_param.txt", dtype=np.float)
for i in param:
    raw = []
    for j in range(aist.shape[0]):
        if aist[j][4] == i:
            raw.append(aist[j][3])
    print(i, sum(raw)/len(raw))
exit(0)


# cal eval_param --- H
param = [0, 1, 2, 3]
aist = np.loadtxt("aist_param.txt", dtype=np.float)
for i in param:
    raw = []
    for j in range(aist.shape[0]):
        if aist[j][4] == i:
            raw.append(aist[j][3])
    print(i, sum(raw)/len(raw))
exit(0)

# cal eval_param --- A
param = [0, 1, 2, 3]
aist = np.loadtxt("aist_u_param.txt", dtype=np.float)
for i in param:
    raw = []
    for j in range(aist.shape[0]):
        if aist[j][3] == i:
            raw.append(aist[j][2])
    print(i, sum(raw)/len(raw))
exit(0)

# cal eval_param --- F
param = [0, 1, 2, 3]
aist = np.loadtxt("aist_eval_param.txt", dtype=np.float)
for i in param:
    raw = []
    for j in range(aist.shape[0]):
        if aist[j][3] == i:
            raw.append(aist[j][2])
    print(i, sum(raw)/len(raw))
exit(0)

# cal eval_model
cat = [0, 1, 2, 3]
aist = np.loadtxt("aist_eval_module.txt", dtype=np.float)
for i in cat:
    raw = []
    com = []
    for j in range(aist.shape[0]):
        if aist[j][3] == i:
            raw.append(aist[j][2])
    # print(raw)
    print(i, sum(raw)/len(raw))

exit(0)
# cal Regression Trees
cat = [1, 2, 3, 7]
aist = np.loadtxt("cart.txt", dtype=np.float)
final_com = []
for i in cat:
    raw_mae = []
    raw_mse = []
    com = []
    for j in range(aist.shape[0]):
        if aist[j][1] == i:
            raw_mae.append(aist[j][2])
            raw_mse.append(aist[j][3])
            com.append(int(aist[j][0]))
    # print(com)
    final_com.append(com)
    print(i, sum(raw_mae)/len(raw_mae), sum(raw_mse)/len(raw_mse))

print(final_com)
exit(0)


y = [1, 2, 3, 4, 8, 50, 51, 52, 53, 54, 55, 30, 32, 33, 34, 35]
z = [i-1 for i in y]

print(z)
exit(0)
x = [i for i in range(1, 78)]
x = set(x)
c1_mae = {0, 5, 6, 7, 71, 21, 22, 23, 24, 27, 31}
c1_mse = {0, 5, 6, 7, 21, 22, 23, 27, 31}
print(x - c1_mae)
print(x- c1_mse)

# cal mae- loss
cat = [1, 2, 3, 7]
aist = np.loadtxt("deepcrime_mae.txt", dtype=np.float)
final_com = []
for i in cat:
    raw = []
    com = []
    for j in range(aist.shape[0]):
        if aist[j][1] == i:
            raw.append(aist[j][2])
            com.append(int(aist[j][0]))
    # print(com)
    final_com.append(com)
    print(i, sum(raw)/len(raw))

print(final_com)
exit(0)


# aist_mae, MiST, aist_mse, mist_mse
c1_mae = {0, 5, 6, 7, 71, 21, 22, 23, 24, 27, 31}
c2_mae = {66, 68, 70, 7, 71, 42, 43, 48, 22, 23, 24}
c3_mae = {66, 68, 70, 7, 71, 42, 43, 22, 24, 28}
c4_mae = {70, 7, 71, 42, 22, 24, 25, 26, 28, 29}

c1_mse = {0, 5, 6, 7, 21, 22, 23, 27, 31}
c2_mse = {65, 66, 68, 70, 7, 42, 43, 22, 24, 28}
c3_mse = {65, 66, 68, 70, 7, 42, 43, 24, 28}
c4_mse = {70, 42, 22, 24, 25, 26, 28, 29}

cat = 1
x = np.loadtxt("geoman_eval_mae.txt", dtype=np.int)
com = []
for i in range(x.shape[0]):
    if x[i][1] == cat:
        com.append(x[i][0])
x = set(com)
print(x)
print(c1_mae - x)
print(x - c1_mae)
exit(0)

"""
order: aist, mist, deepcrime
Cat 1 MAE:
aist = {0, 5, 6, 7, 71, 21, 22, 23, 24, 27, 31}
mist = {0, 5, 6, 7, 71, 21, 22, 23, 24, 27, 31} 
       {0, 5, 6, 7, 21, 22, 23, 24, 27, 31} 71


MSE:
aist = {0, 5, 6, 7, 21, 22, 23, 27, 31}
mist = {0, 5, 6, 7, 21, 22, 23, 27, 31} 

-------------------------------------------------------------------------------------------------------------
Cat 2 MAE:
aist = {66, 68, 70, 7, 71, 42, 43, 48, 22, 23, 24}  
mist = {66, 68, 70, 7, 71, 42, 43, 48, 22, 23, 24}
       {65, 70, 42, 48, 22, 23, 24, 28}  66 68 7 71 43 


MSE:
{65, 66, 68, 70, 7, 42, 43, 22, 24, 28} 
{65, 68, 70, 7, 42, 48, 22, 23, 24, 28} 

-----------------------------------------------------------------------------------------------------------------
Cat 3 MAE:
aist = {66, 68, 70, 7, 71, 42, 43, 22, 24, 28}
mist = {66, 68, 70, 7, 71, 42, 43, 22, 24, 28}
       {66, 68, 70, 7, 42, 43, 22, 24, 28} 71


aist = {65, 66, 68, 70, 7, 42, 43, 24, 28}
mist = {65, 66, 68, 70, 7, 42, 43, 24, 28}
----------------------------------------------------------------------------------------------------------------- 
Cat 7 MAE:
aist = {70, 7, 71, 42, 22, 24, 25, 26, 28, 29}
mist = {70, 71, 7, 42, 22, 24, 25, 26, 28, 29}
dc   = {70, 42, 22, 24, 25, 26, 28, 29}  {71, 7}



MSE
aist = {70, 42, 22, 24, 25, 26, 28, 29}
mist = {70, 42, 22, 24, 25, 26, 28, 29}
ha = {70, 42, 22, 24, 25, 26, 28, 29}

-----------------------------------------------------------------------------------------------------------------
"""