import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

data = []
label = []
np.random.seed(0)
#以原点为圆心,半径为1的圆把散点分成红蓝两部分，并加入如随机噪声
for i in range(150):
    x1 = np.random.uniform(-1, 1)
    x2 = np.random.uniform(0, 2)
    data.append([np.random.normal(x1, 0.1), np.random.normal(x2, 0.1)])
    if x1**2 + x2**2 <= 1:
        label.append(0)
    else:
        label.append(1)
data = np.hstack(data).reshape(-1, 2)
label = np.hstack(label).reshape(-1, 1)
plt.scatter(data[:,0], data[:,1], c=np.squeeze(label),
           cmap="RdBu", vmin=-.2, vmax=1.2, edgecolor="white")
plt.show()

#获取一层神经网络边上的权重
def get_weight(shape, lambda1):
    #生成一个变量
    var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    return var

x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))
keep_prob = tf.placeholder(tf.float32)
#定义每一层网络中节点的个数
layer_dimension = [2, 10, 5, 3, 1]
#神经网络的层数
n_layers = len(layer_dimension)
#这个变量维护前进传播时最深层的节点，开始的时候是输入层
cur_layer = x
#当前层的节点
in_dimension = layer_dimension[0]
#通过一个循环来生成5层全连接的神经网络结构
for i in range(1, n_layers):
    #layer_dimension[i]为下一层的节点个数
    out_dimension = layer_dimension[i]
    #生成当前层中权重的变量，并将这个变量的L2正则化损失加入计算图上的集合
    weight = get_weight([in_dimension, out_dimension], 0.003)
    bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))
    wx_plus_b = tf.matmul(cur_layer, weight) + bias
    # 调用dropout功能
    wx_plus_b = tf.nn.dropout(wx_plus_b, keep_prob)
    #使用eLU函数
    cur_layer = tf.nn.elu(wx_plus_b)
    #进入下一层之前将一层的节点个数更新为当前层节点个数
    in_dimension = layer_dimension[i]

y = cur_layer
#在定义神经网络前向传播的同时已经将所有的L2正则化损失加入了图上的集合,
#这里只需要计算刻画模型在训练数据上表现的损失函数
mse_loss = tf.reduce_mean(tf.square(y_ - y))
train_op = tf.train.AdamOptimizer(0.001).minimize(mse_loss)

#使用dropout
TRAINING_STEPS = 40000
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(TRAINING_STEPS):
        sess.run(train_op, feed_dict={x: data, y_: label, keep_prob: 0.7})
        if i % 2000 == 0:
            print("After %d steps， loss: %f" % (i,
                    sess.run(mse_loss, feed_dict={x: data, y_: label, keep_prob: 1.0})))
        #画出训练后的分割曲线
        xx, yy = np.mgrid[-1.2:1.2:0.1, -0.2:2.2:.01]
        grid = np.c_[xx.ravel(), yy.ravel()]
        probs = sess.run(y, feed_dict={x: grid, keep_prob: 1.0})
        probs = probs.reshape(xx.shape)
plt.scatter(data[:,0], data[:,1], c=np.squeeze(label),
           cmap="RdBu", vmin=-.2, vmax=1.2, edgecolor="white")
plt.contour(xx, yy, probs, levels=[.5], cmap="Greys", vmin=0, vmax=.1)
plt.show()


