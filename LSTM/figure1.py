#画单个的分割图
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
from seg_figure import figure

list2=[[85],[67],[58],[63]]
name=[0,1,2,3]
for i in range(len(list2)):
    df = pd.read_csv(r"E:/毕业论文/Data/ZZData/552/"+str(name[i])+".csv")
    dqhh = df["DQHH"]
    a = dqhh[-dqhh.duplicated()].values
    f1 = df[df["DQHH"] == a[0]]
    host = host_subplot(111, axes_class=AA.Axes)
    figure(host,f1,list2=list2[i])
    plt.draw()
    plt.show()

