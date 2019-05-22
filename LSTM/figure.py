#画整体的分割图
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
from seg_figure import figure

# df=pd.read_csv(r"E:/毕业论文/Data/ZZData/552/552.csv")
df=pd.read_csv("smooth_552.csv")
#去除空值
df.dropna(axis=0,how='any',inplace=True)
df.reset_index(drop=True)
dqhh=df["DQHH"]
a=dqhh[-dqhh.duplicated()].values
#修改a[]里面的参数，画对应环的图像
f1=df[df["DQHH"]==a[0]]

# list1=[3641, 4306,4336, 4965,4996, 5675,5709, 6051]
# list2=[3726, 4403, 5054, 5772]

host = host_subplot(111, axes_class=AA.Axes)
figure(host,f1)
plt.title("552环掘进阶段合并后的光滑图")
plt.draw()
plt.show()

