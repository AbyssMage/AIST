import numpy as np
import pandas as pd
import shapefile as shp
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figure
fig, ax = plt.subplots(figsize=(6, 5))
com = [5, 6, 7, 8, 32, 24, 28, 33, 31, 28, 22, 23, 27, 24, 26, 21, 25, 29, 30, 15, 19]  # 25 18, 19, 15,
col =[7, 8, 32]
shp_path = "E:/aist/Boundaries/chicago.shp"
sf = shp.Reader(shp_path)
print(sf.record())
# plt.figure(figsize=(9, 4))
frameon=False

id = 0
y_lim = (41.7687,42.03) # latitude
x_lim = (-87.7989, -87.5572) # longitudeplot_map(sf, x_lim, y_lim)
y_lim = None
x_lim = None
for shape in sf.shapeRecords():
    if int(shape.record[2]) in com:
        x = [i[0] for i in shape.shape.points[:]]
        y = [i[1] for i in shape.shape.points[:]]
        plt.plot(x, y, 'k', linewidth=0.7)
        x0 = np.mean(x) - .015
        y0 = np.mean(y) - 0.001
        plt.text(x0, y0, shape.record[2], fontsize=14)
    if int(shape.record[2]) in col:
        shape_ex = shape.shape
        x_lon = np.zeros((len(shape_ex.points), 1))
        y_lat = np.zeros((len(shape_ex.points), 1))
        for ip in range(len(shape_ex.points)):
            x_lon[ip] = shape_ex.points[ip][0]
            y_lat[ip] = shape_ex.points[ip][1]
        if int(shape.record[2]) == 7:
            ax.fill(x_lon, y_lat, 'darkseagreen')
        if int(shape.record[2]) == 8:
            ax.fill(x_lon, y_lat, 'cornflowerblue')
        if int(shape.record[2]) == 32:
            ax.fill(x_lon, y_lat, 'c')
        if int(shape.record[2]) == 24:
            ax.fill(x_lon, y_lat, 'orchid')


frame1 = plt.gca()
frame1.axes.get_xaxis().set_visible(False)
frame1.axes.get_yaxis().set_visible(False)
ax.axis('off')
plt.gca().set_axis_off()
# plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
# plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
fig.savefig('map.png', transparent=True)
plt.show()
fig.savefig('E:/aist/graphs/map.svg', format='svg', dpi=1200)

