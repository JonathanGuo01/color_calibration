# -*- coding:utf-8 -*-
from data_proc import data_proc
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib as mpl

input_rgb, output_rgb = data_proc()
trainX = input_rgb/255.
trainY = output_rgb/255.

x_train, x_test, y_train, y_test = train_test_split(trainX, trainY, train_size=0.7, random_state=1)
print(trainX.shape,trainY.shape)
print(x_train.shape,y_train.shape)

sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32,[None,3])

# layer 1
width_1 = 18
w_1 = tf.Variable(tf.ones(shape=(3,width_1),dtype=tf.float32)/10.)
b_1 = tf.Variable(tf.ones(shape=(1,width_1),dtype=tf.float32)/10.)
y_1 = tf.nn.relu(tf.matmul(x,w_1)+b_1)

# layer 2
width_2 = 15
w_2 = tf.Variable(tf.ones(shape=(width_1,width_2),dtype=tf.float32)/10.)
b_2 = tf.Variable(tf.ones(shape=(1,width_2),dtype=tf.float32)/10.)
y_2 = tf.nn.relu(tf.matmul(y_1,w_2)+b_2)

# layer 3
width_3 = 3
w_3 = tf.Variable(tf.ones(shape=(width_2,width_3),dtype=tf.float32)/10.)
b_3 = tf.Variable(tf.ones(shape=(1,width_3),dtype=tf.float32)/10.)
y_3 = tf.nn.relu(tf.matmul(y_2,w_3)+b_3)

y_output = y_3

# trainning
y_std = tf.placeholder(tf.float32,[None,3])
mse = tf.reduce_mean((y_std-y_output)**2)
# mse = np.mean((y_std-y_output)**2,dtype=np.float32)

train_step =tf.train.GradientDescentOptimizer(learning_rate=0.03).minimize(loss=mse)
#18,15,0.03,10000,mse = 0.02239402

tf.global_variables_initializer().run()

mse_plot = []
for i in range(10000):
    train_step.run({x:x_train,y_std:y_train})
    mse_eval = mse.eval({x:x_train,y_std:y_train})
    mse_plot.append(mse_eval)
    print(i,": ",mse_eval)

# estimate
accuracy_loss = tf.reduce_mean((y_std-y_output)**2)
print(accuracy_loss.eval({x:x_test,y_std:y_test}))

# plot
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False
plt.plot(mse_plot)
plt.xlabel(u'迭代次数', fontsize=12)
plt.ylabel(u'均方差（MSE）', fontsize=12)
plt.title(u'训练误差曲线', fontsize=14)
plt.grid(True)
plt.show()






