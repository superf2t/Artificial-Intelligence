import tensorflow as tf

class Linear_Model:

    def __init__(self):
        self.build()

    def __call__(self):
        return self.train_op

    def build(self):
        self.k = tf.Variable(0., name="k")
        self.b = tf.Variable(0., name="b")

        y_predict = self.k * x + self.b

        # 损失函数，均方误差
        self.loss = tf.reduce_mean(tf.square(y - y_predict))
        # 梯度下降优化损失
        self.train_op = tf.train.GradientDescentOptimizer(0.1).minimize(self.loss)


with tf.Session() as sess:
    #初始化变量
    x = tf.placeholder(tf.float32, shape=(1))
    y = tf.placeholder(tf.float32, shape=(1))

    model = Linear_Model()
    train_op = model()
    loss_output = model.loss

    loss = tf.placeholder(tf.float32, shape=(1))
    loss_summary = tf.summary.scalar("loss", tf.squeeze(loss))

    # y = 1 *x + 2
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("./log", tf.get_default_graph())
    for i in range(100):
        # one echo
        _, loss1 = sess.run([train_op, loss_output], feed_dict={x: [1.], y: [3]})
        _, loss2 = sess.run([train_op, loss_output], feed_dict={x: [2.], y: [4]})

        loss_one_echo = (loss1 + loss2) / 2
        summary = sess.run(loss_summary, feed_dict={loss: [loss_one_echo]})
        writer.add_summary(summary, i)

        print ("echo {} loss: {}".format(i, round(loss_one_echo, 3)))
    writer.close()

    vars = tf.global_variables()
    for var in vars:
        print(var.name, " : ", var.eval())
