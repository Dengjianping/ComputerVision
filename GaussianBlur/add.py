import tensorflow as tf
import numpy as np

w = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)

x = tf.placeholder(tf.float32)
model = w*x + b
y = tf.placeholder(tf.float32)

loss = tf.reduce_sum(tf.square(model - y))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

x_train = [1,2,3,4]
y_train = [0, -1, -2, -3]

init = tf.global_variables_initializer()
session = tf.Session()

session.run(init)

for i in range(200):
    data_feed = {x: x_train, y: y_train}
    result = session.run(train, {x: x_train, y: y_train})
    if (i % 10 == 0):
        print(session.run(w), session.run(b))
        # print(session.run(loss))

session.close()