import tensorflow as tf
#创建图
t = tf.add(8, 9)
#创建会话
with tf.Session() as session:
    #启动后进行计算
    result = session.run(t)
print(t)
print(result)

