import tensorflow as tf

#创建一个变量，初始化为标量2.0
scale = tf.Variable(2.0, name="scale")
#创建一个常量张量
age = tf.constant(10.0)
#创建替代任意操作的张量
input = tf.placeholder(tf.float32)
output = tf.multiply(scale * age, input)
with tf.Session() as session:
    #初始化变量
    init = tf.global_variables_initializer()
    session.run(init)
    print (session.run([output], feed_dict={input:[7.]}))
