# coding=utf-8

import numpy as np
import tensorflow as tf
from getdata import get_train_data
from lstm_net import lstm
import matplotlib.pyplot as plt
time_step=5
batch_size=50
rnn_unit =10 # 隐层神经元的个数
lstm_layers = 5  # 隐层层数
input_size = 5
output_size = 1
lr = 0.001  # 学习率
#xloc 表示需要哪些x作为输入
#yloc 表示需要哪个y作为标签
xloc=[0,1,2,3,4]
yloc=4

batch_index, train_x, train_y = get_train_data(batch_size, time_step, xloc,yloc)
# ————————————————训练模型————————————————————
with tf.name_scope('inputs'):
    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size],name="X")
    Y = tf.placeholder(tf.float32, shape=[None, time_step,output_size],name="Y")
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')


pred, _ = lstm(X,input_size,rnn_unit,lstm_layers,keep_prob)
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])),name="loss")
    tf.summary.scalar("loss", loss)
with tf.name_scope('train'):
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)
saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
plt.ion()
losss=[]
with tf.Session() as sess:
    writer = tf.summary.FileWriter("logs/", sess.graph)
    sess.run(tf.global_variables_initializer())

    for i in range(200):  # 这个迭代次数，可以更改，越大预测效果会更好，但需要更长时间
        for step in range(len(batch_index) - 1):
            _, loss_ = sess.run([train_op, loss], feed_dict={X: train_x[batch_index[step]:batch_index[step + 1]],
                                                             Y: train_y[batch_index[step]:batch_index[step + 1]],
                                                             keep_prob: 0.5})
        losss.append(loss_)
        plt.cla()
        ax.plot(losss)
        plt.xlabel("iters")
        plt.ylabel("error")
        plt.pause(0.01)
        print("Number of iterations:", i, " loss:", loss_)

        plt.savefig("error.jpg")
    print("model_save: ", saver.save(sess, 'model_save2\\modle.ckpt'))
    print("The train has finished")




