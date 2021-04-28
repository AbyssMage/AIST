import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

"""
File Conventions: t_a = trend attention, re_a = recent time step attention, d_a = daily time step attention
w_a = weekly time step attention, f_a = feature attention, r_a = region attetnion
ci = crime category
"""
target_region = 71


#  Generate Data for attention to TRENDS(recent, daily, trend) for a target region
c1 = torch.from_numpy(np.loadtxt("Heatmap/t_a_" + str(target_region) + "_1.txt", dtype=np.float))
c2 = torch.from_numpy(np.loadtxt("Heatmap/t_a_" + str(target_region) + "_2.txt", dtype=np.float))
c3 = torch.from_numpy(np.loadtxt("Heatmap/t_a_" + str(target_region) + "_3.txt", dtype=np.float))
c7 = torch.from_numpy(np.loadtxt("Heatmap/t_a_" + str(target_region) + "_7.txt", dtype=np.float))
trend_att = torch.stack([c1, c2, c3, c7], dim=0)

# save the TRENDS to generate a seperate heatmap for them
trend_att_tem = torch.stack([c1, c2, c3, c7], dim=0).mean(dim=1).numpy()
df_t_att = pd.DataFrame(trend_att_tem)
df_t_att.columns = ['Recent', 'Daily', 'Weekly']
df_t_att.index = ['C1', 'C2', 'C3', 'C7']


#  --------------------------------------------------------------------------------------------------------------------------------------
#  Generate Data for attention to RECENT TIMESTEPS for a target region
c1 = torch.from_numpy(np.loadtxt("Heatmap/re_a_" + str(target_region) + "_1.txt", dtype=np.float))
c2 = torch.from_numpy(np.loadtxt("Heatmap/re_a_" + str(target_region) + "_2.txt", dtype=np.float))
c3 = torch.from_numpy(np.loadtxt("Heatmap/re_a_" + str(target_region) + "_3.txt", dtype=np.float))
c7 = torch.from_numpy(np.loadtxt("Heatmap/re_a_" + str(target_region) + "_7.txt", dtype=np.float))
re_att = torch.stack([c1, c2, c3, c7], dim=0)

#  --------------------------------------------------------------------------------------------------------------------------------------
#  Generate Data for attention to DAILY TIMESTEPS for a target region
c1 = torch.from_numpy(np.loadtxt("Heatmap/d_a_" + str(target_region) + "_1.txt", dtype=np.float))
c2 = torch.from_numpy(np.loadtxt("Heatmap/d_a_" + str(target_region) + "_2.txt", dtype=np.float))
c3 = torch.from_numpy(np.loadtxt("Heatmap/d_a_" + str(target_region) + "_3.txt", dtype=np.float))
c7 = torch.from_numpy(np.loadtxt("Heatmap/d_a_" + str(target_region) + "_7.txt", dtype=np.float))
d_att = torch.stack([c1, c2, c3, c7], dim=0)

#  --------------------------------------------------------------------------------------------------------------------------------------
#  Generate Data for attention to WEEKLY TIMESTEPS for a target region
c1 = torch.from_numpy(np.loadtxt("Heatmap/w_a_" + str(target_region) + "_1.txt", dtype=np.float))
c2 = torch.from_numpy(np.loadtxt("Heatmap/w_a_" + str(target_region) + "_2.txt", dtype=np.float))
c3 = torch.from_numpy(np.loadtxt("Heatmap/w_a_" + str(target_region) + "_3.txt", dtype=np.float))
c7 = torch.from_numpy(np.loadtxt("Heatmap/w_a_" + str(target_region) + "_7.txt", dtype=np.float))
w_att = torch.stack([c1, c2, c3, c7], dim=0)

#  --------------------------------------------------------------------------------------------------------------------------------------
#  Generate Data for attention to TOTAL TIMESTEPS timesteps for a target region

# Multiply the trends contribution of trends (RECENT) with respective time step
tem1 = trend_att[:, :, 0:1].repeat(1, 1, re_att.shape[2])
re_att = (re_att * tem1).mean(dim=1)
# Multiply the trends contribution of trends (DAILY) with respective time step
tem2 = trend_att[:, :, 1:2].repeat(1, 1, d_att.shape[2])
d_att = (d_att * tem2).mean(dim=1)
# Multiply the trends contribution of trends (WEEKLY) with respective time step
tem3 = trend_att[:, :, 2:3].repeat(1, 1, w_att.shape[2])
w_att = (w_att * tem3).mean(dim=1)

# Generate data for attention to TOTAL TIMESTEPS (WITH TRENDS CONDTRIBUTION MULTIPLIED)
ts_att = torch.zeros((4, 10))
ts_att[:, 0:1] = w_att[:, 0:1]
ts_att[:, 1:2] = d_att[:, 0:1]
ts_att[:, 2:3] = w_att[:, 1:2]
ts_att[:, 3:4] = d_att[:, 1:2]
ts_att[:, 4:5] = d_att[:, 2:3] + w_att[:, 2:3]
ts_att[:, 5:8] = re_att[:, 0:3]
ts_att[:, 8:9] = d_att[:, 3:4]
ts_att[:, 9:10] = re_att[:, 3:4]
ts_att = ts_att.numpy()

df_ts_att = pd.DataFrame(ts_att)
df_ts_att.columns = ['T-119', 'T-95','T-77','T-65','T-35','T-15','T-11','T-7','T-5','T-3']
df_ts_att.index = ['C1', 'C2', 'C3', 'C7']

#  --------------------------------------------------------------------------------------------------------------------------------------
#  Generate heatmaps for attention to NEIGHBOR REGIONS for a target region
c1 = torch.from_numpy(np.loadtxt("Heatmap/r_a_" + str(target_region) + "_1.txt", dtype=np.float).mean(axis=0))
c2 = torch.from_numpy(np.loadtxt("Heatmap/r_a_" + str(target_region) + "_2.txt", dtype=np.float).mean(axis=0))
c3 = torch.from_numpy(np.loadtxt("Heatmap/r_a_" + str(target_region) + "_3.txt", dtype=np.float).mean(axis=0))
c7 = torch.from_numpy(np.loadtxt("Heatmap/r_a_" + str(target_region) + "_7.txt", dtype=np.float).mean(axis=0))
r_att = torch.stack([c1, c2, c3, c7], dim=0).numpy()

# Generate a dataframe
df_r_att = pd.DataFrame(r_att)
# Set the columns and indices of the dataframe
# df_r_att.columns = ['R7', 'R24', 'R28', 'R32', 'R8']
# df_r_att.columns = ['R18', 'R19', 'R23', 'R26', 'R29', 'R25']
df_r_att.columns = ['R70', 'R71', 'R73', 'R74', 'R75', 'R72']
df_r_att.index = ['C1', 'C2', 'C3', 'C4']

#  --------------------------------------------------------------------------------------------------------------------------------------
#  Generate heatmaps for attention to FEATURES for a target region
c1 = torch.from_numpy(np.loadtxt("Heatmap/f_a_" + str(target_region) + "_1.txt", dtype=np.float).mean(axis=0))
c2 = torch.from_numpy(np.loadtxt("Heatmap/f_a_" + str(target_region) + "_2.txt", dtype=np.float).mean(axis=0))
c3 = torch.from_numpy(np.loadtxt("Heatmap/f_a_" + str(target_region) + "_3.txt", dtype=np.float).mean(axis=0))
c7 = torch.from_numpy(np.loadtxt("Heatmap/f_a_" + str(target_region) + "_7.txt", dtype=np.float).mean(axis=0))
f_att = torch.stack([c1, c2, c3, c7], dim=0).numpy()

# Generate a dataframe
df_f_att = pd.DataFrame(f_att)
# Set the columns and indices of the dataframe
df_f_att.columns = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7',  'F8',  'F9',  'F10',  'F11', 'F12']
df_f_att.index = ['C1', 'C2', 'C3', 'C7']


#  --------------------------------------------------------------------------------------------------------------------------------------
# Create subplots for all your heatmaps to stack in
fig = plt.figure(figsize=(14, 4))
fig.subplots_adjust(wspace=0.07)
ax1 = fig.add_subplot(1, 4, 1)  # If your canvas has p1 rows and p2 columns then you must position your fig at index p3
ax2 = fig.add_subplot(1, 4, 2)
ax3 = fig.add_subplot(1, 4, 3)
ax4 = fig.add_subplot(1, 4, 4)

a1 = sns.heatmap(data=df_r_att, ax=ax1, cmap="YlGnBu", cbar=False)
for _, spine in a1.spines.items():
    spine.set_visible(True)

a2 = sns.heatmap(data=df_ts_att, ax=ax2, cmap="YlGnBu", cbar=False)
for _, spine in a2.spines.items():
    spine.set_visible(True)

a3 = sns.heatmap(data=df_f_att, ax=ax3, cmap="YlGnBu", cbar=False)
for _, spine in a3.spines.items():
    spine.set_visible(True)

a4 = sns.heatmap(data=df_t_att, ax=ax4, cmap="YlGnBu", cbar=True)
for _, spine in a4.spines.items():
    spine.set_visible(True)

a1.set_yticklabels(a1.get_yticklabels(), rotation=-0)
a1.set_xticklabels(a1.get_xticklabels(), rotation=90)
a1.tick_params(axis='x', labelsize=15)
a1.tick_params(axis='y', labelsize=13)
a1.tick_params(axis='both', which='both', length=0)

a2.tick_params(axis='both', which='both', length=0)
a2.set_yticklabels(a2.get_yticklabels(), visible=False)
a2.tick_params(axis='both', which='both', length=0)
a2.tick_params(axis='x', labelsize=15)

a3.tick_params(axis='both', which='both', length=0)
a3.set_yticklabels(a2.get_yticklabels(), visible=False)
a3.set_xticklabels(a3.get_xticklabels(), rotation=90)
a3.tick_params(axis='both', which='both', length=0)
a3.tick_params(axis='x', labelsize=15)

a4.tick_params(axis='both', which='both', length=0)
a4.set_yticklabels(a2.get_yticklabels(), visible=False)
a4.set_xticklabels(a4.get_xticklabels(), rotation=90)
a4.tick_params(axis='x', labelsize=15)


# plt.subplots_adjust(bottom=0.24, left=0.07, right=0.94)
plt.margins(0, 0)
plt.tight_layout()
plt.show()
fig.savefig('E:/aist/graphs/r' + str(target_region) + '.svg', format='svg', dpi=1200)

