from random import randint

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from utils import tf_display_prediction

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.Session()

X = tf.placeholder(tf.float32, [None, 784])
Y_ = tf.placeholder(tf.float32, [None, 10])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

init = tf.global_variables_initializer()

sess.run(init)
y = tf.matmul(X, W) + b
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y_, logits=y))

optimizer = tf.train.GradientDescentOptimizer(0.003)  # petit pas
train_step = optimizer.minimize(cross_entropy)

for _ in range(1000):
    batch = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={X: batch[0], Y_: batch[1]})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval(session=sess, feed_dict={X: mnist.test.images, Y_: mnist.test.labels}))

tf_display_prediction(randint(0, 55000), mnist, sess, y, X)
