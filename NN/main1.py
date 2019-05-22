import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False


path0=r"E:/毕业论文/Data/train.csv"
path1=r"E:/毕业论文/Data/test.csv"
df0=pd.read_csv(path0)
df1=pd.read_csv(path1)
train=df0[["TJSD","DPZS","LJZS","ZTJL","DPNJ"]].values
test=df1[["TJSD","DPZS","LJZS","ZTJL","DPNJ"]].values

def normalization(x):
    c=x.shape[1]
    mean=np.mean(x,axis=0)
    std=np.std(x,axis=0)
    for i in range(c):
        x[:,i]=(x[:,i]-mean[i])/std[i]
    return x

train_X=normalization(train[:,[0,1,2,3]])
test_X=normalization(test[:,[0,1,2,3]])

learning_rate=0.5
n_input=4
n_label=1
n_hidden1=5
n_hidden2=5


x=tf.placeholder(tf.float32,[None,n_input])
y=tf.placeholder(tf.float32,[None,n_label])

weights={
    'h1':tf.Variable(tf.truncated_normal([n_input,n_hidden1],stddev=0.1)),
    'h2':tf.Variable(tf.truncated_normal([n_hidden1,n_hidden2],stddev=0.1)),
    'h3':tf.Variable(tf.truncated_normal([n_hidden2,n_label],stddev=0.1))
    }

biases={
    'h1':tf.Variable(tf.zeros([n_hidden1])),
    'h2':tf.Variable(tf.zeros([n_hidden2])),
    'h3':tf.Variable(tf.zeros([n_label]))
    }

layer_1=tf.nn.sigmoid(tf.add(tf.matmul(x,weights['h1']),biases['h1']))
layer_2=tf.nn.sigmoid(tf.add(tf.matmul(layer_1,weights['h2']),biases['h2']))
y_pred=tf.add(tf.matmul(layer_2,weights['h3']),biases['h3'])
loss=tf.reduce_mean((y_pred-y)**2)
train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

maxEpochs=10
minibatchSize=100

Y=np.reshape(train[:,-1]/1000,(-1,1))
sess=tf.Session()
sess.run(tf.global_variables_initializer())
m=np.int32(Y.shape[0]/minibatchSize)

ax=[]
loss3=[]

for epoch in range(maxEpochs):
    sumloss=0
    for i in range(m):
        x1=train_X[i*minibatchSize:(i+1)*minibatchSize,:]
        y1=Y[i*minibatchSize:(i+1)*minibatchSize,:]
        sess.run(train_step,feed_dict={x:x1,y:y1})
        loss1=sess.run(loss,feed_dict={x:x1,y:y1})
        sumloss+=loss1
    print("Epoch:",'%04d'%(epoch+1),"loss=",sumloss/m)
    ax.append(epoch)
    loss3.append(sumloss/m)



#测试集效果

y_pred1=sess.run(y_pred,feed_dict={x:test_X})
y_pred1=np.reshape(y_pred1,[-1,1])
e=np.reshape(test[:,-1]/1000,[-1,1])
#y:np.reshape(test[:,-1]/1000,(-1,1))
data=np.hstack((y_pred1,e))

print(np.cov(data,rowvar=False))

y_pred2=sess.run(y_pred,feed_dict={x:train_X,y:np.reshape(train[:,-1]/1000,(-1,1))})
fig=plt.figure()
xx=[i for i in range(test.shape[0])]
yy=[i for i in range(train.shape[0])]
ax1=fig.add_subplot(212)
l1,=ax1.plot(xx,test[:,-1],color='g')
l2,=ax1.plot(xx,y_pred1*1000,color='y')
plt.legend([l1,l2],["测试集_刀盘扭矩","预测_刀盘扭矩"])
ax2=fig.add_subplot(211)
l3,=ax2.plot(yy,train[:,-1],color='g')
l4,=ax2.plot(yy,y_pred2*1000,color='y')
plt.legend([l3,l4],["训练集_刀盘扭矩","模型_刀盘扭矩"])
plt.show()
