# libraries
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(5, 5))

height1 = [1.153409003, 0.8946701329, 0.8856704359, 0.9566262814, 0.874652097]
height2 = [0.3727272729, 0.372778041, 0.3722757067, 0.3730705051, 0.3614835233]
height3 = [0.7344195111, 0.719026169, 0.6944157189, 0.7225138995, 0.6910386575]
height7 = [0.3699999944, 0.3493950446, 0.3404296132, 0.3586622131, 0.339946472]

current = height2
cat = 2

bars = ('AIST$\mathregular{_r}$', 'AIST$\mathregular{_d}$', 'AIST$\mathregular{_w}$', 'AIST$\mathregular{_l}$', 'AIST')
ax.set_xticklabels(ax.get_xticks(), fontsize=16)
# ax.set_yticklabels(ax.get_yticks(), fontsize=15)
y_pos = np.arange(len(bars))
plt.bar(y_pos, current, color=(0.5, 0.1, 0.5, 0.6), width=0.8)
plt.xticks(y_pos, bars, rotation=0)
plt.yticks(fontsize=16)

plt.xlabel('Module', fontsize=16)
plt.ylabel('MAE', fontsize=16)
l_limit = min(current) - 0.09
h_limit = max(current) + 0.001
plt.ylim(l_limit, h_limit)
plt.xticks(y_pos, bars)
# plt.subplots_adjust(bottom=0.18, left=0.15)

fig.tight_layout()
fig.savefig('E:/aist/eps/tmodule' + str(cat) + '.eps', format='eps', dpi=350)
plt.show()
