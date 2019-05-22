import numpy as np
import pandas as pd

# df =pd.read_csv(r"E:/毕业论文/Data/all_data.csv")
# f1=df[df["DQHH"].isin([552,554,555,563,564,565])].reset_index(drop=True)
# f2=f1[["DQHH","TJSD","DPZS","LJZS","ZTJL","DPNJ"]]
# f2.to_csv(r"E:/毕业论文/Data/data.csv",index=False)
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

df1 =pd.read_csv(r"E:/毕业论文/Data/ZZData/552/stable.csv")
df1=smooth(df1)

df2 =pd.read_csv(r"E:/毕业论文/Data/ZZData/554/stable.csv")
df2=smooth(df2)
df3 =pd.read_csv(r"E:/毕业论文/Data/ZZData/555/stable.csv")
df3=smooth(df3)
df4 =pd.read_csv(r"E:/毕业论文/Data/ZZData/563/stable.csv")
df4=smooth(df4)
df5 =pd.read_csv(r"E:/毕业论文/Data/ZZData/564/stable.csv")
df5=smooth(df5)
df6 =pd.read_csv(r"E:/毕业论文/Data/ZZData/565/stable.csv")
df6=smooth(df6)
a=[df1,df2,df3,df4,df5]
pd.concat(a).reset_index(drop=True).to_csv(r"E:/毕业论文/Data/train.csv", index=False)
df6.to_csv(r"E:/毕业论文/Data/test.csv",index=False)

