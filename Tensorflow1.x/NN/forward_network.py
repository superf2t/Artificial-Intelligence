import tensorflow as tf

w1 = tf.Variable(tf.random_normal((2, 3), stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal((3, 1), stddev=1, seed=1))

x = tf.constant([[0.7, 0.9]])   # 1 * 2
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

session = tf.Session()
# session.run(w1.initializer)
# session.run(w2.initializer)
init_operation = tf.global_variables_initializer()
session.run(init_operation)

print(session.run(y))
session.close()
