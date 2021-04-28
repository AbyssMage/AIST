# libraries
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------------------------------------------------------------
# Graph for Tw
fig, ax = plt.subplots(figsize=(4, 4))
height1 = [0.874652097, 0.9035767196, 0.8906750193, 0.8554943774]
height2 = [0.3614835233, 0.3730004066, 0.3726522456, 0.3719226247]
height3 = [0.6910386575, 0.7042789191, 0.692613643, 0.6870386104]
height7 = [0.339946472, 0.3436230315, 0.3468160454, 0.3475153164]

current = height7
cat = 7

bars = ('3', '4', '5', '6')
y_pos = np.arange(len(bars))
plt.bar(y_pos, current, color=(0.5, 0.1, 0.5, 0.6))
ax.set_xticklabels(ax.get_xticks(), fontsize=16)
plt.yticks(fontsize=16)

# ax.set_yticklabels(ax.get_yticks(), fontsize=14)
plt.xlabel('Number of Weekly Time Steps ($T_w$)', size=16)
plt.ylabel('MAE', size=16)
l_limit = min(current) - 0.005
h_limit = max(current) + 0.001
plt.ylim(l_limit, h_limit)
plt.xticks(y_pos, bars)
plt.subplots_adjust(bottom=0.167, left=0.252, right=0.779)

# fig.savefig('E:/aist/graphs/param_tw' + str(cat) + '.svg', format='svg', dpi=1200)
# fig.savefig('E:/aist/graphs/param_tw' + str(cat) + '.png', dpi=300)
# ax.set_rasterized(True)
fig.savefig('E:/aist/eps/param_tw' + str(cat) + '.eps', format='eps', dpi=300)
plt.show()
exit()
# --------------------------------------------------------------------------------------------------------------------------------------
# Graph for Td
fig, ax = plt.subplots(figsize=(4, 4))
height1 = [0.8851330162, 0.8905746223, 0.874652097, 0.9138985711]
height2 = [0.3733297367, 0.3731322504, 0.3614835233, 0.3732936384]
height3 = [0.7052562013, 0.7035774248, 0.6910386575, 0.6969214384]
height7 = [0.3415447523, 0.3456838036, 0.339946472, 0.3443174415]

current = height7
cat = 7

bars = ('12', '16', '20', '24')
y_pos = np.arange(len(bars))
plt.bar(y_pos, current, color=(0.5, 0.1, 0.5, 0.6))
ax.set_xticklabels(ax.get_xticks(), fontsize=16)
plt.yticks(fontsize=16)

# ax.set_yticklabels(ax.get_yticks(), fontsize=14)
plt.xlabel('Number of Daily Time Steps ($T_d$)', size=16)
plt.ylabel('MAE', size=16)
l_limit = min(current) - 0.01
h_limit = max(current) + 0.003
plt.ylim(l_limit, h_limit)
plt.xticks(y_pos, bars)
plt.subplots_adjust(bottom=0.15, left=0.295, right=0.81)

# fig.savefig('E:/aist/graphs/param_td' + str(cat) + '.svg', format='svg', dpi=1200)
fig.tight_layout()
fig.savefig('E:/aist/eps/param_td' + str(cat) + '.eps', format='eps', dpi=300)
plt.show()
exit()

#  16/3/21 --------------------------------------------------------------------------------------------------------------------------------------
# Graph for Q
"""fig, ax = plt.subplots(figsize=(4, 4))
height1 = [0.8744302088, 0.874652097, 0.9173720255, 0.8869951379]
height2 = [0.3725936691, 0.3614835233, 0.3727391658, 0.372854521]
height3 = [0.6995490024, 0.6910386575, 0.7095668501, 0.7007313193]
height7 = [0.3438302857, 0.339946472, 0.356475577, 0.3413088742]

current = height7
cat = 7

bars = ('36', '40', '44', '48')
y_pos = np.arange(len(bars))
plt.bar(y_pos, current, color=(0.5, 0.1, 0.5, 0.6))
ax.set_xticklabels(ax.get_xticks(), fontsize=16)
plt.yticks(fontsize=16)
# ax.set_yticklabels(ax.get_yticks(), fontsize=14)
plt.xlabel('Dimension of Query, Key ($d_q, d_k$)', size=16)
plt.ylabel('MAE', size=16)
l_limit = min(current) - 0.01
h_limit = max(current) + 0.003
plt.ylim(l_limit, h_limit)
plt.xticks(y_pos, bars)

fig.savefig('E:/aist/graphs/param_Q' + str(cat) + '.svg', format='svg', dpi=1200)
plt.show()
exit()"""
#  16/3/21 --------------------------------------------------------------------------------------------------------------------------------------
# Graph for Tr
"""fig, ax = plt.subplots(figsize=(4, 4))
height1 = [0.8962054329, 0.874652097, 0.8651544541, 0.8694014684]
height2 = [0.3718254913, 0.3614835233, 0.3723726296, 0.3729078486]
height3 = [0.7152926968, 0.6910386575, 0.7000729873, 0.6992899172]
height7 = [0.3477911317, 0.339946472, 0.3555015611, 0.3417704877]

current = height7
cat = 7

bars = ('16', '20', '24', '28')
y_pos = np.arange(len(bars))
plt.bar(y_pos, current, color=(0.5, 0.1, 0.5, 0.6))
ax.set_xticklabels(ax.get_xticks(), fontsize=16)
plt.yticks(fontsize=16)
# ax.set_yticklabels(ax.get_yticks(), fontsize=14)
plt.xlabel('Number of Recent Time Steps (T)', size=16)
plt.ylabel('MAE', size=16)
l_limit = min(current) - 0.01
h_limit = max(current) + 0.005
plt.ylim(l_limit, h_limit)
plt.xticks(y_pos, bars)
# plt.subplots_adjust(bottom=0.18, left=0.15)

fig.savefig('E:/aist/graphs/param_T' + str(cat) + '.svg', format='svg', dpi=1200)
plt.show()
exit()"""
#  16/3/21 --------------------------------------------------------------------------------------------------------------------------------------
# Graph for F
"""fig, ax = plt.subplots(figsize=(4, 4))
height1 = [0.8732379619, 0.874652097, 0.8800823203, 0.8856096147]
height2 = [0.3707296314, 0.3614835233, 0.3726204822, 0.3729368452]
height3 = [0.7236393066, 0.6910386575, 0.7247051089, 0.7062138145]
height7 = [0.3639567682, 0.339946472, 0.355835832, 0.3556431049]

current = height7
cat = 7

bars = ('6', '8', '10', '12')
y_pos = np.arange(len(bars))
plt.bar(y_pos, current, color=(0.5, 0.1, 0.5, 0.6))
ax.set_xticklabels(ax.get_xticks(), fontsize=16)
plt.yticks(fontsize=16)
# ax.set_yticklabels(ax.get_yticks(), fontsize=14)
plt.xlabel("Output dimension of hGAT, fGAT (F)", size=16)
plt.ylabel('MAE', size=16)
l_limit = min(current) - 0.005
h_limit = max(current) + 0.001
plt.ylim(l_limit, h_limit)
plt.xticks(y_pos, bars)
# plt.subplots_adjust(bottom=0.18, left=0.15)

fig.savefig('E:/aist/graphs/param_F' + str(cat) + '.svg', format='svg', dpi=1200)
plt.show()
exit()"""

#  --------------------------------------------------------------------------------------------------------------------------------------
# Graph for A
"""fig, ax = plt.subplots(figsize=(4, 4))

height1 = [0.8920629081, 0.874652097, 0.9062115618, 0.8928279434]
height2 = [0.3716033093, 0.3614835233, 0.3733171226, 0.3734892091]
height3 = [0.7210241835, 0.6910386575, 0.7068274181, 0.7135030107]
height7 = [0.3567811692, 0.339946472, 0.349411824, 0.349071603]

current = height7
cat = 7

bars = ('22', '30', '38', '46')
y_pos = np.arange(len(bars))
plt.bar(y_pos, current, color=(0.5, 0.1, 0.5, 0.6))
ax.set_xticklabels(ax.get_xticks(), fontsize=16)
plt.yticks(fontsize=16)

plt.xlabel('Dimension of Location Attention (A)', fontsize=16)
plt.ylabel('MAE', fontsize=16)

l_limit = min(current) - 0.01
h_limit = max(current) + 0.003
plt.ylim(l_limit, h_limit)
plt.xticks(y_pos, bars)

# plt.subplots_adjust(bottom=0.18, left=0.16, right = 0.84)

fig.savefig('E:/aist/graphs/param_a' + str(cat) + '.svg', format='svg', dpi=1200)
plt.show()
exit()"""
#  --------------------------------------------------------------------------------------------------------------------------------------
# Graph for H
# fig, ax = plt.figure(figsize=(4, 3))
"""fig, ax = plt.subplots(figsize=(4, 4))

height1 = [0.9135956776, 0.90523914, 0.8746520970285498, 0.9368189016]
height2 = [0.3740930485, 0.3718728524, 0.36148352331171435, 0.3727274133]
height3 = [0.7162428834, 0.722525165, 0.6910386574634957, 0.7783486693]
height7 = [0.3487125257, 0.3549512367, 0.3399464720120978, 0.3444082542]

current = height7
cat = 7

bars = ('24', '32', '40', '48')
y_pos = np.arange(len(bars))
plt.bar(y_pos, current, color=(0.5, 0.1, 0.5, 0.6))
ax.set_xticklabels(ax.get_xticks(), fontsize=16)
plt.yticks(fontsize=16)
# ax.set_yticklabels(ax.get_yticks(), fontsize=14)
plt.xlabel('Dimension of Hidden State (H)', size=16)
plt.ylabel('MAE', size=16)
l_limit = min(current) - 0.01
h_limit = max(current) + 0.003
plt.ylim(l_limit, h_limit)
plt.xticks(y_pos, bars)
plt.subplots_adjust(bottom=0.18, left=0.15)

fig.savefig('E:/aist/graphs/param_h' + str(cat) + '.svg', format='svg', dpi=1200)
plt.show()
exit()"""

#  --------------------------------------------------------------------------------------------------------------------------------------
# Graphs for F
"""fig = plt.figure(figsize=(4, 3))
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
plt.show()"""

#  --------------------------------------------------------------------------------------------------------------------------------------
# Graphs for A
"""base = 0.8746520970285498 / 1.0921199789655114
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
plt.show()"""
#  --------------------------------------------------------------------------------------------------------------------------------------
# Graph for different saptial modules: 15/3/21
"""fig, ax = plt.subplots(figsize=(4, 4))

height1 = [0.8935698165, 0.8914884112, 0.8835086795, 0.874652097]
height2 = [0.3752025971, 0.3732298141, 0.3726518517, 0.3614835233]
height3 = [0.7169978335, 0.7144736712, 0.6927084606, 0.6910386575]
height7 = [0.3557148401, 0.3429943211, 0.3581500146, 0.339946472]

current = height7
cat = 7

bars = ('AIST$\mathregular{_g}$', 'AIST$\mathregular{_h}$', 'AIST$\mathregular{_f}$', 'AIST')
ax.set_xticklabels(ax.get_xticks(), fontsize=15)
# ax.set_yticklabels(ax.get_yticks(), fontsize=15)
y_pos = np.arange(len(bars))
plt.bar(y_pos, current, color=(0.5, 0.1, 0.5, 0.6), width=0.8)
plt.xticks(y_pos, bars, rotation=0)
plt.yticks(fontsize=14)

plt.xlabel('Module', fontsize=15)
plt.ylabel('MAE', fontsize=15)
l_limit = min(current) - 0.006
h_limit = max(current) + 0.001
plt.ylim(l_limit, h_limit)
plt.xticks(y_pos, bars)
plt.subplots_adjust(bottom=0.18, left=0.15)

fig.savefig('E:/aist/graphs/module' + str(cat) + '.svg', format='svg', dpi=1200)
plt.show()"""

#  --------------------------------------------------------------------------------------------------------------------------------------
# Graph for different saptial modules: 15/3/21
fig, ax = plt.subplots(figsize=(5, 5))

height1 = [1.153409003, 0.8946701329, 0.8856704359, 0.9566262814, 0.874652097]
height2 = [0.3727272729, 0.372778041, 0.3722757067, 0.3730705051, 0.3614835233]
height3 = [0.7344195111, 0.719026169, 0.6944157189, 0.7225138995, 0.6910386575]
height7 = [0.3699999944, 0.3493950446, 0.3404296132, 0.3586622131, 0.339946472]

current = height1
cat = 1

bars = ('AIST$\mathregular{_r}$', 'AIST$\mathregular{_d}$', 'AIST$\mathregular{_w}$', 'AIST$\mathregular{_l}$', 'AIST')
ax.set_xticklabels(ax.get_xticks(), fontsize=16)
# ax.set_yticklabels(ax.get_yticks(), fontsize=15)
y_pos = np.arange(len(bars))
plt.bar(y_pos, current, color=(0.5, 0.1, 0.5, 0.6), width=0.8)
plt.xticks(y_pos, bars, rotation=0)
plt.yticks(fontsize=16)

plt.xlabel('Module', fontsize=16)
plt.ylabel('MAE', fontsize=16)
l_limit = min(current) - 0.15
h_limit = max(current) + 0.001
plt.ylim(l_limit, h_limit)
plt.xticks(y_pos, bars)
# plt.subplots_adjust(bottom=0.18, left=0.15)

# fig.savefig('E:/aist/graphs/tmodule' + str(cat) + '.svg', format='svg', dpi=1200)
plt.show()
exit()
#  --------------------------------------------------------------------------------------------------------------------------------------
# Graph for different saptial modules: 15/3/21 + another column
"""fig, ax = plt.subplots(figsize=(5, 5))

height1 = [0.8935698165, 0.8914884112, 0.8835086795, 0.8922640991, 0.874652097]
height2 = [0.3752025971, 0.3732298141, 0.3726518517, 0.3740353183, 0.3614835233]
height3 = [0.7169978335, 0.7144736712, 0.6927084606, 0.7116890123, 0.6910386575]
height7 = [0.3557148401, 0.3429943211, 0.3581500146, 0.3494099021, 0.339946472]

current = height2
cat = 2

bars = ('AIST$\mathregular{_g}$', 'AIST$\mathregular{_h}$', 'AIST$\mathregular{_f}$', "AIST$\mathregular{_{f'}}$", 'AIST')
ax.set_xticklabels(ax.get_xticks(), fontsize=16)
# ax.set_yticklabels(ax.get_yticks(), fontsize=15)
y_pos = np.arange(len(bars))
plt.bar(y_pos, current, color=(0.5, 0.1, 0.5, 0.6), width=0.8)
plt.xticks(y_pos, bars, rotation=0)
plt.yticks(fontsize=16)

plt.xlabel('Module', fontsize=16)
plt.ylabel('MAE', fontsize=16)
l_limit = min(current) - 0.006
h_limit = max(current) + 0.001
plt.ylim(l_limit, h_limit)
plt.xticks(y_pos, bars)
# plt.subplots_adjust(bottom=0.18, left=0.15)

fig.savefig('E:/aist/graphs/smodule' + str(cat) + '.svg', format='svg', dpi=1200)
plt.show()"""