import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
n = 5
nfeature = 10
community = [7, 31, 6, 23, 27]

poi = np.loadtxt("poi.txt", dtype=np.int)
e1 = np.ones((n, nfeature), dtype=np.int)
for i in range(n):
    e1[i] = poi[community[i]]
corr_matrix_e1 = np.corrcoef(e1)

x = 2190
c1 = np.ones((n, 4), dtype=np.int)
cat = [1, 2, 3, 7]
for i in range(n):
    for j in range(4):
        c1[i][j] = np.sum(np.loadtxt("data/com_crime/r_" + str(community[i]) + ".txt", dtype=np.int)[cat[j]])
corr_matrix_c1 = np.corrcoef(c1)


x = 2190
t1 = np.ones((n, 2), dtype=np.int)
for i in range(n):
    for j in range(2):
        t1[i][j] = np.sum(np.loadtxt("data/act_ext/taxi_" + str(community[i] + 1) + ".txt", dtype=np.int)[j])
corr_matrix_t1 = np.corrcoef(t1)

df_c = pd.DataFrame(corr_matrix_c1)
df_c.columns = ['R8', 'R32', 'R7', 'R24', 'R28']
df_c.index = ['R8', 'R32', 'R7', 'R24', 'R28']

df_poi = pd.DataFrame(corr_matrix_e1)
df_poi.columns = ['R8', 'R32', 'R7', 'R24', 'R28']
df_poi.index = ['R8', 'R32', 'R7', 'R24', 'R28']

df_t = pd.DataFrame(corr_matrix_t1)
df_t.columns = ['R8', 'R32', 'R7', 'R24', 'R28']
df_t.index = ['R8', 'R32', 'R7', 'R24', 'R28']

# Create subplots for all your heatmaps to stack in
fig = plt.figure(figsize=(10, 3))
fig.subplots_adjust(wspace=0.07)
# fig, ax3 = plt.subplots(figsize=(6, 5))

ax1 = fig.add_subplot(1, 3, 1)  # If your canvas has p1 rows and p2 columns then you must position your fig at index p3
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)

a1 = sns.heatmap(data=df_c, ax=ax1, cmap="YlGnBu", cbar=True)
for _, spine in a1.spines.items():
    spine.set_visible(True)
a2 = sns.heatmap(data=df_poi, ax=ax2, cmap="YlGnBu", cbar=True)
for _, spine in a2.spines.items():
    spine.set_visible(True)

a3 = sns.heatmap(data=df_t, ax=ax3, cmap="YlGnBu", cbar=True)
for _, spine in a3.spines.items():
    spine.set_visible(True)

a1.set_yticklabels(a1.get_yticklabels(), rotation=90)
a1.set_xticklabels(a1.get_xticklabels())
a1.tick_params(axis='x', labelsize=13)
a1.tick_params(axis='y', labelsize=13)
a1.tick_params(axis='both', which='both', length=0)

a2.set_yticklabels(a2.get_yticklabels())
a2.tick_params(axis='x', labelsize=13)
a2.tick_params(axis='y', labelsize=13)
a2.tick_params(axis='both', which='both', length=0)

a3.set_yticklabels(a3.get_yticklabels())
a3.set_xticklabels(a3.get_xticklabels())
a3.tick_params(axis='x', labelsize=13)
a3.tick_params(axis='y', labelsize=13)
a3.tick_params(axis='both', which='both', length=0)

# plt.subplots_adjust(bottom=0.24, left=0.07, right=0.94)

plt.margins(0, 0)
plt.tight_layout()
plt.show()
fig.savefig('E:/aist/graphs/hgat_cor.svg', format='svg', dpi=1200)