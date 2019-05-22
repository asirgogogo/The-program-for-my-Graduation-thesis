import numpy as np # 快速操作结构数组的工具
import matplotlib.pyplot as plt  # 可视化绘制
from sklearn.linear_model import Lasso,LassoCV,LassoLarsCV
import pandas as pd
from sklearn import preprocessing
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False


path0=r"E:/毕业论文/Data/train.csv"
path1=r"E:/毕业论文/Data/test.csv"
df0=pd.read_csv(path0)
df1=pd.read_csv(path1)
train=df0[["TJSD","DPZS","LJZS","ZTJL","DPNJ"]].values
test=df1[["TJSD","DPZS","LJZS","ZTJL","DPNJ"]].values
train[:,[0,1,2,3]]=preprocessing.scale(train[:,[0,1,2,3]])
train[:,-1]=train[:,-1]/1000
test[:,[0,1,2,3]]=preprocessing.scale(test[:,[0,1,2,3]])


model=LassoCV()
model.fit(train[:,[0,1,2,3]],train[:,-1])
print('系数矩阵:\n',model.coef_)
print('线性回归模型:\n',model)
print('最佳的alpha：',model.alpha_)
predicted = model.predict(test[:,[0,1,2,3]])
predicted=np.reshape(predicted,[-1,1])
e=np.reshape(test[:,-1]/1000,[-1,1])
data=np.hstack((predicted,e))

print(np.cov(data,rowvar=False))

l1,=plt.plot(predicted*1000,c='r',ls='-')
l2,=plt.plot(test[:,-1],c='b',ls=':')
plt.legend([l1,l2],["Predict Values","True Values"])
# 绘制x轴和y轴坐标

plt.ylabel("刀盘扭矩")

# 显示图形
plt.show()
