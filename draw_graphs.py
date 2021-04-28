import numpy as np
import torch
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter

#  --------------------------------------------------------------------------------------------------------------------------------------
# time series for TEMPORAL correlation - 16/3
plt.rc('font', family='serif')
target_region = 7
fig, ax = plt.subplots(figsize=(6.5, 5.001))

r_8 = np.loadtxt("data/com_crime/r_" + str(target_region) + ".txt", dtype=np.int)
#  load the crime types for whom we want to find the temporal correlation
loaded_data = np.ones((3, r_8.shape[1]))
loaded_data[0] = r_8[0]
loaded_data[1] = r_8[1]
loaded_data[2] = r_8[4]

# For the purpose of checking
loaded_data = torch.from_numpy(loaded_data)

cat = 2
cat0 = loaded_data[cat].view(365, -1)
print(cat0.shape)
print(cat0)
print(cat0.sum(dim=0))
print(cat0.shape)
exit()



loaded_data = loaded_data.T

# df = pd.DataFrame(loaded_data[30:55, :])
df = pd.DataFrame(loaded_data[30:60, :])

# For the purpose of checking
loaded_data = loaded_data[30:60]
loaded_data = torch.from_numpy(loaded_data).T

cat = 2
cat0 = loaded_data[cat].view(6, -1)
print(cat0)
print(cat0.sum(dim=0))
print(cat0.shape)
exit()


df.columns = ['C0: Deceptive Practice', 'C1: Theft', 'C5: Robbery']

sns.lineplot(data=df)
xlbls = ['12AM', '8AM', '4PM',
         '12AM', '8AM', '4PM',
         '12AM', '8AM', '4PM',
         '12AM', '8AM', '4PM',
         '12AM', '8AM', '4PM',
         '12AM'
         ]
plt.xticks((np.arange(0, 31, step=2)), xlbls, fontsize=8)
# ax.set_xticklabels(ax.get_xticks())
ax.set_ylabel("Number of Crimes", fontsize=18)

loaded_data = np.ones((3, r_8.shape[1]))
loaded_data = np.negative(loaded_data)
loaded_data = loaded_data.T

ax2 = ax.twiny()
ax2.xaxis.set_ticks_position('bottom')
ax2.xaxis.set_label_position('bottom')
ax2.set_xticklabels(ax2.get_xticks(), fontsize=30)
ax2.spines['bottom'].set_position(('outward', 20))

df2 = pd.DataFrame(loaded_data[0:30, :])

sns.lineplot(data=df2, legend=False, ax=ax2)
ax2.set_ylim(bottom=-0.1)

xlbls2 = ["Jan6", "Jan7", "Jan8", "Jan9", "Jan10", "Jan11"]
plt.xticks((np.arange(0, 31, step=6)), xlbls2, fontsize=15)
# ax2.set_xticklabels(ax.get_xticks(), fontsize=20)
ax2.set_xlabel("Date", fontsize=18)

plt.yticks((np.arange(0, 10, step=4)), fontsize=14)
plt.subplots_adjust(bottom=0.186)
plt.savefig('Figure/tem_cor.png', dpi=300)
plt.show()
fig.savefig('E:/aist/graphs/tem_cor.svg', format='svg', dpi=1200)
exit()

# time series for SPATIAL correlation: 18 March 2021
# ---------------------------------------------------------------------------------------------------------------------------------------
plt.rc('font', family='serif')

target_cat = 0
fig, ax = plt.subplots(figsize=(6, 5.001))

#  load the regions for whom we want to find the spatial correlation
r_8 = np.loadtxt("data/com_crime/r_7.txt", dtype=np.int)
r_32 = np.loadtxt("data/com_crime/r_31.txt", dtype=np.int)
r_7 = np.loadtxt("data/com_crime/r_6.txt", dtype=np.int)
r_24 = np.loadtxt("data/com_crime/r_23.txt", dtype=np.int)

# final data: loaded_data
loaded_data = np.ones((4, r_8.shape[1]), dtype=np.int)
loaded_data[0] = r_8[target_cat]
loaded_data[1] = r_32[target_cat]
loaded_data[2] = r_7[target_cat]
loaded_data[3] = r_24[target_cat]
loaded_data = loaded_data.T

# Generate Dataframe
# df = pd.DataFrame(loaded_data[12:43, :3])
df = pd.DataFrame(loaded_data[30:60, :3])
df.columns = ['R8', 'R32', 'R7']

# Select fonts
csfont = {'fontname': 'Comic Sans MS'}
hfont = {'fontname': 'Times New Roman'}

# Create line plots
a = sns.lineplot(data=df)

a.set_xticklabels(a.get_xticks(), fontsize=15)
plt.xlabel('Date', fontsize=18)
plt.ylabel('Number of Crimes', **hfont, fontsize=18)
# x_formatter = FixedFormatter(["Jan3", "Jan4", "Jan5", "Jan6", "Jan7", "Jan8"])
x_formatter = FixedFormatter(["Jan6", "Jan7", "Jan8", "Jan9", "Jan10", "Jan11"])
x_locator = FixedLocator([0, 6, 12, 18, 24, 30])
a.xaxis.set_major_formatter(x_formatter)
a.xaxis.set_major_locator(x_locator)
plt.yticks((np.arange(0, 5, step=2)), fontsize=14)
fig.savefig('E:/aist/graphs/sp_cor.svg', format='svg', dpi=1200)
plt.show()
exit()
#  --------------------------------------------------------------------------------------------------------------------------------------
# time series for TAXI INFLOW OUTFLOW correlation: 17 March
plt.rc('font', family='serif')
# plt.rc('xtick', labelsize='x-small')
# plt.rc('ytick', labelsize='x-small')
target_region = 7
cat = 1  # 0, 1, 4
fig, ax = plt.subplots(figsize=(6, 5.001))
csfont = {'fontname': 'Comic Sans MS'}
hfont = {'fontname': 'Times New Roman'}

r_8 = np.loadtxt("data/com_crime/r_" + str(target_region) + ".txt", dtype=np.int)
t_8 = np.loadtxt("data/ext/taxi" + str(target_region) + ".txt", dtype=np.int)

#  load the crime types for whom we want to find the temporal correlation
loaded_data_c = np.ones((1, r_8.shape[1]))

loaded_data_c[0] = r_8[cat]

loaded_data_c = loaded_data_c.T
loaded_data_t = np.ones((2, t_8.shape[1]))
loaded_data_t[0] = t_8[0]
loaded_data_t[1] = t_8[1]
loaded_data_t = loaded_data_t.T

df1 = pd.DataFrame(loaded_data_t[30:60, :], )
df1.columns = ['Taxi Inflow', 'Taxi Outflow']
# ax.plot(df1) BuPu Dark2 Dark2 RdGy YlGn

sns.lineplot(data=df1, ax=ax, palette='BuPu')
ax.set_xlabel("Date", fontsize=18)
ax.set_ylabel("Taxi Flow", fontsize=18)
# ax.set_xlabel("Date", **hfont, fontsize=18)
# ax.set_ylabel("Traffic Flow", **hfont, fontsize=18)
x_formatter = FixedFormatter(["Jan6", "Jan7", "Jan8", "Jan9", "Jan10", "Jan11"])
x_locator = FixedLocator([0, 6, 12, 18, 24, 30])
ax.xaxis.set_major_formatter(x_formatter)
ax.xaxis.set_major_locator(x_locator)
ax.set_xticklabels(ax.get_xticks(), fontsize=15)
plt.yticks((np.arange(0, 820, step=200)), fontsize=14)

# twin object for two different y-axis on the sample plot
ax2 = ax.twinx()
ax2.set_xticklabels(ax2.get_xticks(), fontsize=14)

df2 = pd.DataFrame(loaded_data_c[30:60, :])
x_formatter = FixedFormatter(["Jan6", "Jan7", "Jan8", "Jan9", "Jan10", "Jan11"])
x_locator = FixedLocator([0, 6, 12, 18, 24, 30])
ax2.xaxis.set_major_formatter(x_formatter)
ax2.xaxis.set_major_locator(x_locator)

df2.columns = ['C1: Theft']

# ax2.plot(df2, color='red')
sns.lineplot(data=df2, palette=("blue",), ax=ax2)
ax2.set_ylabel("Number of Crimes", fontsize=18)
# ax2.set_ylabel("Crime Distribution", **hfont, fontsize=18)
# plt.yticks((np.arange(0, 5, step=1)), fontsize=14)
plt.yticks((np.arange(0, 10, step=2)), fontsize=14)

# ax.figure.legend(['Inflow', 'Outflow'])
# plt.subplots_adjust(left=0.136)
fig.savefig('E:/aist/graphs/feat_cor2.svg', format='svg', dpi=1200)
plt.show()
exit()

#  --------------------------------------------------------------------------------------------------------------------------------------

# time series for TEMPORAL correlation
target_region = 7
fig, ax = plt.subplots(figsize=(6, 5.001))

r_8 = np.loadtxt("data/com_crime/r_" + str(target_region) + ".txt", dtype=np.int)

#  load the crime types for whom we want to find the temporal correlation
loaded_data = np.ones((3, r_8.shape[1]))
loaded_data[0] = r_8[0]
loaded_data[1] = r_8[1]
loaded_data[2] = r_8[4]
loaded_data = loaded_data.T

df = pd.DataFrame(loaded_data[30:60, :])
df.columns = ['C1', 'C2', 'C3']

csfont = {'fontname': 'Comic Sans MS'}
hfont = {'fontname': 'Times New Roman'}

a = sns.lineplot(data=df)

a.set_xticklabels(a.get_xticks(), fontsize=15)
plt.xlabel('Time', **hfont, fontsize=20)
plt.ylabel('Number of Crimes', **hfont, fontsize=20)
x_formatter = FixedFormatter(["Jan6", "Jan7", "Jan8", "Jan9", "Jan10", "Jan11"])
x_locator = FixedLocator([0, 6, 12, 18, 24, 30])
a.xaxis.set_major_formatter(x_formatter)
a.xaxis.set_major_locator(x_locator)
plt.yticks((np.arange(0, 10, step=2)), fontsize=15)
plt.show()

#  --------------------------------------------------------------------------------------------------------------------------------------
# time series for TAXI INFLOW OUTFLOW correlation
target_region = 7
fig, ax = plt.subplots(figsize=(6, 5.001))

t_8 = np.loadtxt("data/ext/taxi" + str(target_region) + ".txt", dtype=np.int)
loaded_data = np.ones((2, t_8.shape[1]))
loaded_data[0] = t_8[0]
loaded_data[1] = t_8[1]
loaded_data = loaded_data.T

df = pd.DataFrame(loaded_data[30:60, :])
df.columns = ['Taxi Inflow', 'Taxi Outflow']

a = sns.lineplot(data=df)

csfont = {'fontname': 'Comic Sans MS'}
hfont = {'fontname': 'Times New Roman'}

a.set_xticklabels(a.get_xticks(), fontsize=15)
plt.xlabel('Time', **hfont, fontsize=20)
plt.ylabel('Taxi flow count', **hfont, fontsize=20)
x_formatter = FixedFormatter(["Jan6", "Jan7", "Jan8", "Jan9", "Jan10", "Jan11"])
x_locator = FixedLocator([0, 6, 12, 18, 24, 30])
a.xaxis.set_major_formatter(x_formatter)
a.xaxis.set_major_locator(x_locator)
plt.yticks((np.arange(100, 900, step=100)), fontsize=12)
plt.show()
