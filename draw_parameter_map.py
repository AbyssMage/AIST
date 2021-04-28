import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

fig = plt.figure(figsize=(14, 4))
ax1 = fig.add_subplot(2, 4, 1)  # If your canvas has p1 rows and p2 columns then you must position your fig at index p3
ax2 = fig.add_subplot(2, 4, 2)
ax3 = fig.add_subplot(2, 4, 3)
ax4 = fig.add_subplot(2, 4, 4)

h_height1 = [0.9135956776, 0.90523914, 0.8746520970285498, 0.9368189016]
h_height2 = [0.3740930485, 0.3718728524, 0.36148352331171435, 0.3727274133]
h_height3 = [0.7162428834, 0.722525165, 0.6910386574634957, 0.7783486693]
h_height7 = [0.3487125257, 0.3549512367, 0.3399464720120978, 0.3444082542]


bars = ('24', '32', '40', '48')
y_pos = np.arange(len(bars))
plt.bar(y_pos, h_height1, color=(0.5, 0.1, 0.5, 0.6))
ax1.set_xticklabels(ax1.get_xticks(), fontsize=16)
ax1.set_yticklabels(ax1.get_yticks(), fontsize=16)

plt.xlabel('Dimension of Hidden State (H)', size=16)
plt.ylabel('MAE', size=16)
l_limit = min(h_height1) - 0.01
h_limit = max(h_height1) + 0.01
plt.ylim(l_limit, h_limit)
plt.xticks(y_pos, bars)
plt.subplots_adjust(bottom=0.18, left=0.15)

# fig.savefig('E:/aist/graphs/param_h' + str(cat) + '.svg', format='svg', dpi=1200)
plt.show()
