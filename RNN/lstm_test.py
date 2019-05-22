# coding=utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
import tensorflow as tf
from getdata import get_test_data
from lstm_net import lstm

time_step=10
batch_size=50
rnn_unit =10 # 隐层神经元的个数
lstm_layers = 5  # 隐层层数
input_size = 5
output_size = 1
lr = 0.001 # 学习率
#xloc 表示需要哪些x作为输入
#yloc 表示需要哪个y作为标签
xloc=[0,1,2,3,4]
yloc=5




# ————————————————预测模型————————————————————

X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
mean, std, test_x, test_y = get_test_data(xloc,yloc,time_step)
pred, _ = lstm(X,input_size,rnn_unit,lstm_layers,keep_prob)
saver = tf.train.Saver(tf.global_variables())
with tf.Session() as sess:
    # 参数恢复
    module_file = tf.train.latest_checkpoint('model_save2')
    saver.restore(sess, module_file)
    test_predict = []
    for step in range(len(test_x) - 1):
        prob = sess.run(pred, feed_dict={X: [test_x[step]], keep_prob: 1})
        predict = prob.reshape((-1))
        test_predict.extend(predict)
    test_y = np.array(test_y) * std[yloc] + mean[yloc]
    test_predict = np.array(test_predict) * std[yloc] + mean[yloc]
    predicted = np.reshape(test_predict, [-1, 1])
    e = np.reshape(test_y[:len(test_predict)], [-1, 1])
    data = np.hstack((predicted, e))

    print(np.cov(data, rowvar=False))
    acc = np.average(np.abs(test_predict - test_y[:len(test_predict)]) / test_y[:len(test_predict)])  # 偏差程度
    print("The accuracy of this predict:", acc)
    # 以折线图表示结果
    plt.figure()
    l1,=plt.plot(list(range(len(test_predict))), test_predict, color='b',ls='-')
    l2,=plt.plot(list(range(3000,3000+len(test_y))), test_y, color='r',ls=':')
    plt.legend([l1,l2],["Predict Value","True Value"])
    plt.ylim(2000,6000)
    plt.ylabel("刀盘扭矩值")
    plt.title("对刀盘扭矩——预测")
    plt.show()

