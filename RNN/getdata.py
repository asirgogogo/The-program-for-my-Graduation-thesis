import pandas as pd
import numpy as np
# ——————————————————导入数据——————————————————————

path0=r"E:/毕业论文/Data/train.csv"
df0=pd.read_csv(path0,engine='python',encoding="GB2312")
train_data=df0[["TJSD","DPZS","LJZS","ZTJL","DPNJ","DPNJ1"]].values
path1=r"E:/毕业论文/Data/test.csv"
df1=pd.read_csv(path1,engine='python',encoding="GB2312")
test_data=df1[["TJSD","DPZS","LJZS","ZTJL","DPNJ","DPNJ1"]].values
# 获取训练集
def get_train_data(batch_size, time_step, xloc,yloc):
    batch_index = []
    normalized_train_data = ( train_data- np.mean(train_data, axis=0)) / np.std(train_data, axis=0)  # 标准化
    train_x, train_y = [], []  # 训练集
    for i in range(len(normalized_train_data) - time_step):
        if i % batch_size == 0:
            batch_index.append(i)
        x = normalized_train_data[i:i + time_step, xloc]
        y = normalized_train_data[i:i + time_step, yloc,np.newaxis]
        #np.newaxis
        train_x.append(x.tolist())
        train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data) - time_step))
    return batch_index, train_x, train_y


# 获取测试集
def get_test_data(xloc,yloc,time_step):
    mean = np.mean(test_data, axis=0)
    std = np.std(test_data, axis=0)
    normalized_test_data = (test_data - mean) / std  # 标准化
    size = (len(normalized_test_data) + time_step - 1) // time_step # 有size个sample
    test_x, test_y = [], []
    for i in range(size - 1):
        x = normalized_test_data[i * time_step:(i + 1) * time_step, xloc]
        y = normalized_test_data[i * time_step:(i + 1) * time_step, yloc]
        test_x.append(x.tolist())
        test_y.extend(y)
    test_x.append((normalized_test_data[(i + 1) * time_step:, xloc]).tolist())
    test_y.extend((normalized_test_data[(i + 1) * time_step:, yloc]).tolist())
    return mean, std, test_x, test_y

