from .layers import *


def encoder_conv28(x, z_dim):
    with tf.variable_scope('i_net'):
        conv1 = conv2d_lrelu(x, 64, 4, 2)
        conv2 = conv2d_lrelu(conv1, 128, 4, 2)
        conv2 = tf.reshape(conv2, [-1, np.prod(conv2.get_shape().as_list()[1:])])
        fc1 = fc_lrelu(conv2, 1024)
        mean = tf.contrib.layers.fully_connected(fc1, z_dim, activation_fn=tf.identity)
        stddev = tf.contrib.layers.fully_connected(fc1, z_dim, activation_fn=tf.sigmoid)
        stddev = tf.maximum(stddev, 0.01)
        sample = mean + tf.multiply(stddev, tf.random_normal(tf.stack([tf.shape(x)[0], z_dim])))
        return [mean, stddev], sample


def encoder_conv64(x, z_dim):
    with tf.variable_scope('i_net'):
        conv = conv2d_bn_lrelu(x, 64, 4, 2)
        conv = conv2d_bn_lrelu(conv, 128, 4, 2)
        conv = conv2d_bn_lrelu(conv, 192, 4, 2)
        conv = conv2d_bn_lrelu(conv, 256, 4, 2)
        fc = tf.reshape(conv, [-1, np.prod(conv.get_shape().as_list()[1:])])
        fc = fc_lrelu(fc, 1024)
        mean = tf.contrib.layers.fully_connected(fc, z_dim, activation_fn=tf.identity)
        stddev = tf.contrib.layers.fully_connected(fc, z_dim, activation_fn=tf.sigmoid)
        stddev = tf.maximum(stddev, 0.01)
        sample = mean + tf.multiply(stddev, tf.random_normal(tf.stack([tf.shape(x)[0], z_dim])))
        return [mean, stddev], sample


def encoder_conv64small(x, z_dim):
    with tf.variable_scope('i_net'):
        conv = conv2d_bn_lrelu(x, 32, 4, 2)
        conv = conv2d_bn_lrelu(conv, 64, 4, 2)
        conv = conv2d_bn_lrelu(conv, 96, 4, 2)
        conv = conv2d_bn_lrelu(conv, 128, 4, 2)
        fc = tf.reshape(conv, [-1, np.prod(conv.get_shape().as_list()[1:])])
        fc = fc_lrelu(fc, 512)
        mean = tf.contrib.layers.fully_connected(fc, z_dim, activation_fn=tf.identity)
        stddev = tf.contrib.layers.fully_connected(fc, z_dim, activation_fn=tf.sigmoid)
        stddev = tf.maximum(stddev, 0.01)
        sample = mean + tf.multiply(stddev, tf.random_normal(tf.stack([tf.shape(x)[0], z_dim])))
        return [mean, stddev], sample


def encoder_conv64large(x, z_dim):
    with tf.variable_scope('i_net'):
        conv = conv2d_bn_lrelu(x, 128, 4, 2)
        conv = conv2d_bn_lrelu(conv, 256, 4, 2)
        conv = conv2d_bn_lrelu(conv, 384, 4, 2)
        conv = conv2d_bn_lrelu(conv, 512, 4, 2)
        fc = tf.reshape(conv, [-1, np.prod(conv.get_shape().as_list()[1:])])
        fc = fc_lrelu(fc,  2048)
        mean = tf.contrib.layers.fully_connected(fc, z_dim, activation_fn=tf.identity)
        stddev = tf.contrib.layers.fully_connected(fc, z_dim, activation_fn=tf.sigmoid)
        stddev = tf.maximum(stddev, 0.01)
        sample = mean + tf.multiply(stddev, tf.random_normal(tf.stack([tf.shape(x)[0], z_dim])))
        return [mean, stddev], sample


def encoder_fc64(x, z_dim):
    with tf.variable_scope('i_net'):
        fc = tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])
        fc = fc_bn_lrelu(fc, 1024)
        fc = fc_bn_lrelu(fc, 1024)
        fc = fc_bn_lrelu(fc, 1024)
        fc = fc_bn_lrelu(fc, 1024)
        mean = tf.contrib.layers.fully_connected(fc, z_dim, activation_fn=tf.identity)
        stddev = tf.contrib.layers.fully_connected(fc, z_dim, activation_fn=tf.sigmoid)
        stddev = tf.maximum(stddev, 0.01)
        sample = mean + tf.multiply(stddev, tf.random_normal(tf.stack([tf.shape(x)[0], z_dim])))
        return [mean, stddev], sample






