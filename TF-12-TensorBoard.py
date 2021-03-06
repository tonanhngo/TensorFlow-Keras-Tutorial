from tensorflow.examples.tutorials.mnist import input_data
import os
import tensorflow as tf
import sys
import urllib

if sys.version_info[0] >= 3:
  from urllib.request import urlretrieve
else:
  from urllib import urlretrieve

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
        tf.summary.histogram("weights", W_conv1)
        tf.summary.histogram("biases", b_conv1)
        tf.summary.histogram("activations", h_conv1)

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
        # Display the image after max pooling on tensorboard
        h_pool1_image = tf.reshape(h_pool1, [-1, 14, 14, 1])
        tf.summary.image('conv1', h_pool1_image, 4)

    # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        x_conv2 = tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
        h_conv2 = tf.nn.relu(x_conv2 + b_conv2)
        tf.summary.histogram("weights", W_conv2)
        tf.summary.histogram("biases", b_conv2)
        tf.summary.histogram("activations", h_conv2)

    # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
        # Display the image after max pooling on tensorboard
        h_pool2_image = tf.reshape(h_pool2, [-1, 7, 7, 1])
        tf.summary.image('conv2', h_pool2_image, 4)

    # After 2 rounds of downsampling, our 28x28 image
    # is down to 7x7 with 64 feature maps.
    with tf.name_scope('fc1'):
        h_pool_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        W_fc1 = weight_variable([7*7*64, 1024])
        b_fc1 = bias_variable([1024])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)
        tf.summary.histogram("weights", W_fc1)
        tf.summary.histogram("biases", b_fc1)
        tf.summary.histogram("activations", h_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the features to 10 classes, one for each digit
    with tf.name_scope('fc-classify'):
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])
        y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

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
        train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy, name='train_step')

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

    # Get sprite and labels file for the embedding projector
    GITHUB_URL ='https://raw.githubusercontent.com/mamcgrath/TensorBoard-TF-Dev-Summit-Tutorial/master/'
    urlretrieve(GITHUB_URL + 'labels_1024.tsv', os.path.join(LOGDIR, 'labels_1024.tsv'))
    urlretrieve(GITHUB_URL + 'sprite_1024.png', os.path.join(LOGDIR, 'sprite_1024.png'))

    # Setup embedding visualization
    embedding = tf.Variable(tf.zeros([1024, 1024]), name="test_embedding")
    assignment = embedding.assign(h_fc1_drop)
    saver = tf.train.Saver()

    config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
    embedding_config = config.embeddings.add()
    embedding_config.tensor_name = embedding.name
    # embedding_config.sprite.image_path = os.path.join(LOGDIR, 'sprite_1024.png')
    # embedding_config.metadata_path = os.path.join(LOGDIR, 'labels_1024.tsv')
    embedding_config.sprite.image_path = 'sprite_1024.png'
    embedding_config.metadata_path = 'labels_1024.tsv'
    # Specify the width and height of a single thumbnail.
    embedding_config.sprite.single_image_dim.extend([28, 28])
    tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)

    # Do the training.
    for i in range(1100):
        batch = mnist.train.next_batch(100)
        if i % 5 == 0:
            summary = sess.run(merged, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            writer.add_summary(summary, i)
        if i % 100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
            print("Step %d, Training Accuracy %g" % (i, float(train_accuracy)))
        if i % 500 == 0:
            sess.run(assignment, feed_dict={x: mnist.test.images[:1024], y_: mnist.test.labels[:1024], keep_prob: 1.0})
            saver.save(sess, os.path.join(LOGDIR, "model.ckpt"), i)
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    # See how model did.
    print("Test Accuracy %g" % sess.run(accuracy, feed_dict={x: mnist.test.images,
                                                             y_: mnist.test.labels,
                                                             keep_prob: 1.0}))

    # Close summary writer
    writer.close()


if __name__ == '__main__':
    main()
