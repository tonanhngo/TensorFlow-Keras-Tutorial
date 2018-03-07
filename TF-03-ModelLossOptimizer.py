from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


def weight_variable(shape):
  """Generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """Generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def main():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # Placeholder that will be fed image data.
    x = tf.placeholder(tf.float32, [None, 784])
    # Placeholder that will be fed the correct labels.
    y_ = tf.placeholder(tf.float32, [None, 10])

    # Define weight and bias.
    W = weight_variable([784, 10])
    b = bias_variable([10])

    # Here we define our model which utilizes the softmax regression.
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    # Define our loss.
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

    # Define our optimizer.
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


if __name__ == '__main__':
    main()