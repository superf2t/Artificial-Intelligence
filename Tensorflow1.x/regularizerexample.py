import tensorflow as tf

#获取一层神经网络边上的权重，并将这个权重的L2正则化损失
#加入名为losses的集合中
def get_weight(shape, lambbda):
    #生成一个变量
    var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    #add_to_collection函数将这个新生成变量的L2正则化损失项加入集合
    #这函数的第一个参数losses是集合的名字，第二个参数是要加入这个集合的内容
    tf.add_to_collection('lossed', tf.contrib.layers.l2_regularizer(lambbda)(var))
    return var
x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))
batch_size = 8