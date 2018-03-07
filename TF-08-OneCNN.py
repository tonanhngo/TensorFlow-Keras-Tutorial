from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

LOGDIR = './tensorflow_logs/mnist_deep'

def weight_variable(shape):
  """Generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name='weight')


def bias_variable(shape):
  """Generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name='bias')


def main():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # Placeholder that will be fed image data.
    x = tf.placeholder(tf.float32, [None, 784], name='x')
    # Placeholder that will be fed the correct labels.
    y_ = tf.placeholder(tf.float32, [None, 10], name='labels')

    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image('input', x_image, 4)

    # Convolutional layer - maps one grayscale image to 32 features.
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        x_conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
        h_conv1 = tf.nn.relu(x_conv1 + b_conv1)

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

    # After downsampling, our 28x28 image is now 14x14
    # with 32 feature maps.
    with tf.name_scope('flatten'):
        h_pool_flat = tf.reshape(h_pool1, [-1, 14*14*32])

    # Map the features to 10 classes, one for each digit
    with tf.name_scope('fc-classify'):
        W_fc2 = weight_variable([14*14*32, 10])
        b_fc2 = bias_variable([10])
        y = tf.matmul(h_pool_flat, W_fc2) + b_fc2

    ###########################

    # Define our loss.
    with tf.name_scope('loss'):
        # Use more numerically stable cross entropy.
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y),
            name='cross_entropy'
        )
        tf.summary.scalar('loss', cross_entropy)

    # Define our optimizer.
    with tf.name_scope('optimizer'):
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy, name='train_step')

    # Define accuracy.
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        correct_prediction = tf.cast(correct_prediction, tf.float32, name='correct_prediction')
        accuracy = tf.reduce_mean(correct_prediction, name='accuracy')
        tf.summary.scalar('accuracy', accuracy)

    # Launch session.
    sess = tf.InteractiveSession()

    # Initialize variables.
    tf.global_variables_initializer().run()

    # Merge all the summary data
    merged = tf.summary.merge_all()

    # Create summary writer
    writer = tf.summary.FileWriter(LOGDIR, sess.graph)

    # Do the training.
    for i in range(1100):
        batch = mnist.train.next_batch(100)
        if i % 5 == 0:
            summary = sess.run(merged, feed_dict={x: batch[0], y_: batch[1]})
            writer.add_summary(summary, i)
        if i % 100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={x:batch[0], y_: batch[1]})
            print("Step %d, Training Accuracy %g" % (i, float(train_accuracy)))
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})

    # See how model did.
    print("Test Accuracy %g" % sess.run(accuracy, feed_dict={x: mnist.test.images,
                                                             y_: mnist.test.labels}))

    # Close summary writer
    writer.close()


if __name__ == '__main__':
    main()
