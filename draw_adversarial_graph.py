# library and dataset
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
target_region = 68
target_cat = 3
src_data = np.loadtxt("aist_eval_module_adv.txt")
# print(src_data.shape)
print(src_data)
tvd = []
jsd = []
for i in range(src_data.shape[0]):
    if src_data[i][0] == target_region and src_data[i][1] == target_cat:
        tvd.append(src_data[i][3])
        jsd.append(src_data[i][4])

# Create data frame
df = pd.DataFrame({'y': tvd, 'x': jsd})
print(df)

# plot with matplotlib
plt.plot('x', 'y', data=df, marker='o', color='mediumvioletred', linestyle=':')
plt.show()
