# libraries
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter

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
df = pd.DataFrame(loaded_data[12:49, :1])
df.columns = ['R1']

# Select fonts
csfont = {'fontname':'Comic Sans MS'}
hfont = {'fontname':'Times New Roman'}

# Create line plots
a = sns.lineplot(data=df)

a.set_xticklabels(a.get_xticks(), fontsize=15)
plt.xlabel('Time', **hfont, fontsize=20)
# plt.ylabel('Number of Crimes', **hfont, fontsize=20)
x_formatter = FixedFormatter(["Sep22", "Sep23", "Sep24", "Sep26", "Sep27", "Sep28", "Sep29"])
x_locator = FixedLocator([0, 6, 12, 18, 24, 30, 36])
a.xaxis.set_major_formatter(x_formatter)
a.xaxis.set_major_locator(x_locator)
# plt.yticks((np.arange(0, 5, step=2)), fontsize=15)
plt.subplots_adjust(bottom=0.18, left=0.19, right=0.80)
plt.show()
exit(0)
# Graph for q_v, q_k
# Prepare your data in a list
fig = plt.figure(figsize=(4, 3))
base = 0.8746520970285498 / 1.146264579695546
pre_height = [1.1454669561074484, 1.1809692709381499, 1.146264579695546, 1.1920165103564542]
height = [0.1, 0.5, 0.3, 0.7, 0.2]

# Name your bars
bars = ('23', '26', '18', '19', '20')
y_pos = np.arange(len(bars))
plt.bar(y_pos, height, color=(0.5, 0.1, 0.5, 0.6))

# Set the xlabels, ylabels, xticks, yticks
# plt.xlabel('Dimension of Query, Key Vectors ($d_q, d_k$)', size=14)
# plt.ylabel('MAE', size=12)
# plt.ylim(0.77, 0.94)
plt.xticks(y_pos, bars)
plt.subplots_adjust(bottom=0.18, left=0.19, right=0.80)
plt.show()

exit(0)
#  --------------------------------------------------------------------------------------------------------------------------------------
# Graph for H
fig = plt.figure(figsize=(4, 3))
base = 0.8746520970285498 / 1.154565273465597
pre_height = [1.175258556099106, 1.2036379208176264, 1.154565273465597, 1.4837573432186177]
height = [(base*h) for h in pre_height]

bars = ('24', '32', '40', '48')
y_pos = np.arange(len(bars))
plt.bar(y_pos, height, color=(0.5, 0.1, 0.5, 0.6))

plt.xlabel('Number of Hidden States (H)', size=14)
plt.ylabel('MAE', size=12)
plt.ylim(0.77, 1.15)
plt.xticks(y_pos, bars)
plt.subplots_adjust(bottom=0.18, left = 0.15)
plt.show()


#  --------------------------------------------------------------------------------------------------------------------------------------
# Graphs for F
fig = plt.figure(figsize=(4, 3))
base = 0.8746520970285498 / 0.9536137937952484
pre_height = [1.142302996551411, 0.9536137937952484, 1.0152490825878022, 0.9923572333455675]
height = [(base*h) for h in pre_height]

bars = ('6', '8', '10', '12')
y_pos = np.arange(len(bars))
plt.bar(y_pos, height, color=(0.5, 0.1, 0.5, 0.6))

plt.xlabel('Dimension of Node Embedding (F)', size=14)
plt.ylabel('MAE', size=12)
plt.ylim(0.77, 1.1)
plt.xticks(y_pos, bars)
plt.subplots_adjust(bottom=0.18, left=0.16)
plt.show()


#  --------------------------------------------------------------------------------------------------------------------------------------
# Graphs for A
base = 0.8746520970285498 / 1.0921199789655114
pre_height = [1.1314045967662303, 1.0921199789655114, 1.1284210359510134, 1.1299376001256323]
height = [(base*h) for h in pre_height]
fig = plt.figure(figsize=(4, 3))

bars = ('22', '30', '38', '46')
y_pos = np.arange(len(bars))
plt.bar(y_pos, height, color=(0.5, 0.1, 0.5, 0.6))


plt.xlabel('Dimension of Location Attention (A)', size=14)
plt.ylabel('MAE', size=12)
plt.ylim(0.8, 0.92)
plt.xticks(y_pos, bars)
plt.subplots_adjust(bottom=0.18, left=0.16, right = 0.84)
plt.show()


#  --------------------------------------------------------------------------------------------------------------------------------------
#  Graphs for effectiveness of different Module
fig = plt.figure(figsize=(4, 3))
base = 1.6985768375896406 / 1.903544717472758
pre_height = [3.6276130548409826, 2.6230745982107813, 2.0156982388943065, 1.903544717472758]
height = [(base*h) for h in pre_height]

bars = ('GAT', 'hGAT', 'hGAT + F', 'hGAT + FGAT')
y_pos = np.arange(len(bars))
plt.bar(y_pos, height, color=(0.5, 0.1, 0.5, 0.6), )
plt.xticks(y_pos, bars, rotation=20)

plt.xlabel('Number of hidden states', size=14)
plt.ylabel('MSE', size=12)
plt.ylim(1.5, 3.3)
plt.xticks(y_pos, bars)
plt.subplots_adjust(left = 0.23, bottom=0.18)
plt.show()
