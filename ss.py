import numpy as np
from pylab import *
from matplotlib import cm#color map 用来上色
data=np.random.rand(3,3)
cmap=cm.Blues#蓝色系
#plt.imshow()函数负责对图像进行处理，并显示其格式，但是不能显示
plt.show(data,interpolation='nearest',cmap=cmap,aspect="auto",vmin=0,vmax=1)#aspect=‘auto‘自动缩放，vimn=0，填充为白色，为1，填充为蓝色
#plt.title(‘heatmap‘)