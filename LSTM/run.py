import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn import preprocessing
from scipy import linalg

df=pd.read_csv(r"E:/毕业论文/Data/data.csv")
#encoding="GB2312"
#去除空值
df.dropna(axis=0,how='any',inplace=True)
df.reset_index(drop=True)
dqhh=df["DQHH"]
hh=dqhh[-dqhh.duplicated()].values

def group(l):
    b=[]
    i=0
    while i<=(len(l)-1):
        a=[l[i]]
        if i<(len(l)-1):
            j=0
            while l[i+j+1]-l[i+j]==1:
                a.append(l[i +j+ 1])
                j=j+1
                if i+j+1>(len(l)-1):
                    break
        i=i+len(a)
        b.append(a)
    return b

def check(Data):
    #Data 只能是一维的
    n = Data.shape[0]
    sum = []
    for i in range(n - 1):
        x1 = np.mean(Data[0:(i + 1)])
        x2 = np.mean(Data[(i + 1):])
        sum.append(np.abs((n - i - 1) * (i + 1) * (x1 - x2)))
    ind = np.argmax(sum)
    return ind

def generate(f1,path1):
    N=f1.values.shape[0]
    left_list=[]
    right_list=[]

    for i in range(N-1):
        if f1.iloc[i]["ZTJL"]>0 and f1.iloc[i]["TJSD"]>0:
            if f1.iloc[i-1]["ZTJL"]<1e-7 or f1.iloc[i-1]["TJSD"]<1e-7:
                left_list.append(i - 1)
            if f1.iloc[i+1]["ZTJL"]<1e-7 or f1.iloc[i+1]["TJSD"]<1e-7:
                right_list.append(i+1)


    normal_data=[]
    zero_data=[]
    normal_list=[]
    zero_list=[i for i in range(len(left_list))]

    if len(left_list)>len(right_list):
        s= len(right_list)
    else:
        s= len(left_list)
    for i in range(s):
        if np.abs(right_list[i]-left_list[i])>=100:
            normal_list.append(i)
            normal_data.append(f1.iloc[left_list[i]+1:right_list[i],:].reset_index(drop=True))
    for i in range(len(normal_data)):
        normal_data[i].to_csv(path1+str(i)+".csv",index=False)

    xx=group(normal_list)
    for i in range(len(xx)):
        if len(xx[i])>1:
            for j in range(len(xx[i])-1):
                q=xx[i][j]
                p=xx[i][j+1]
                zero_data.append(f1.iloc[right_list[q]:left_list[p]+1,:].reset_index(drop=True))
        if i<(len(xx)-1):
            zero_data.append(f1.iloc[right_list[xx[i][-1]]:left_list[xx[i+1][0]]+1,:].reset_index(drop=True))


    stable_point_index=[]

    for i in range(len(normal_data)):
        f2 = normal_data[i]
        dpnj_ind=check(f2['DPNJ'].values)
        ztjl_ind=check(f2['ZTJL'].values)
        tjsd_ind=check(f2['TJSD'].values)
        l=[dpnj_ind,ztjl_ind,tjsd_ind]
        stable_point_index.append(np.min(l))
    transition=[]
    stable=[]
    for i in range(len(stable_point_index)):
        if normal_data[i].ix[0:(stable_point_index[i]+1),:].empty:
            pass
        else:
            transition.append(normal_data[i].ix[0:(stable_point_index[i]+1),:])

        if normal_data[i].ix[(stable_point_index[i]+1):,:].empty:
            pass
        else:
            stable.append(normal_data[i].ix[(stable_point_index[i]+1):,:])
    if len(transition)!=0:
        pd.concat(transition).reset_index(drop=True).to_csv(path1+"transition.csv", index=False)
    if len(stable)!=0:
        pd.concat(stable).reset_index(drop=True).to_csv(path1+"stable.csv", index=False)
    print("合并结束!!!")
    return left_list,right_list,xx,stable_point_index
# 下面是两种平滑方式，对数据进行平均值光滑,x必须是数据框
def avg_smooth(x, num):
    return x.rolling( num, center=True).mean()
def median_smooth(x, num):
    return x.rolling( num, center=True).median()

#num：光滑的长度
#method：光滑的方法
def smooth(data,num=9,method=avg_smooth):
    floor = int(np.floor(num / 2))
    n = data.values.shape[0]
    l = [int(k) for k in range(floor)]
    l1 = [int(n - floor + o) for o in range(floor)]
    l.extend(l1)
    data=method(data,num).drop(l,axis=0).reset_index(drop=True)
    return data

f=open("log.txt","w")
for h in hh:
    path="./ZZData/"+str(int(h))
    os.makedirs(path)
    f1=df[df["DQHH"]==h].reset_index(drop=True)
    f1.to_csv(path+"/"+str(int(h))+".csv",index=False)
    q,w,e,r=generate(f1,path+"/")
    f.write("\t"+"环号:"+str(int(h))+"\n")
    f.write("划分间隔左："+str(q)+"\n")
    f.write("划分间隔右：" + str(w)+"\n")
    f.write("正常抽取区间的索引：" + str(e)+"\n")
    f.write("启动阶段-掘进阶段分段点：" + str(r)+"\n")
    f.write("\n")
f.close()


