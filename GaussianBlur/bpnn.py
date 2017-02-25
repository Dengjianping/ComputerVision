import cv2, math, time
import numpy as np
import tensorflow as tf

def addLayer(inputs, inputSize, outputSize, activation=None):
    weights = tf.Variable(tf.random_normal([inputSize, outputSize]),dtype=tf.float32)
    biases = tf.Variable(tf.zeros([1,outputSize]) + 0.1, dtype=tf.float32)
    # chnage data type
    # weights.dtype = tf.float32

    inputs = tf.cast(inputs, tf.float32)
    linearInput = tf.matmul(inputs, weights) + biases
    if activation is None:
        outputs = linearInput
    else:
        outputs = activation(linearInput)
    return outputs

# --
x_data = np.linspace(-1,1,300)[:, np.newaxis]
print(x_data)
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 +noise

xs = tf.placeholder(tf.float32, [None, 1])
print(xs)
ys = tf.placeholder(tf.float32, [None, 1])

l1 = addLayer(x_data, 1, 10, activation=tf.nn.relu)
prediction = addLayer(11, 10, 1)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()

sess.run(init)

for i in range(1000):
    result = sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        print(result)