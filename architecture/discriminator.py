from .layers import *


def discriminator_conv28(x, reuse=False):
    with tf.variable_scope('d_net') as vs:
        if reuse:
            vs.reuse_variables()
        conv1 = conv2d_lrelu(x, 64, 4, 2)
        conv2 = conv2d_lrelu(conv1, 128, 4, 2)
        conv2 = tf.reshape(conv2, [-1, np.prod(conv2.get_shape().as_list()[1:])])
        fc1 = fc_lrelu(conv2, 1024)
        fc2 = tf.contrib.layers.fully_connected(fc1, 1, activation_fn=tf.identity)
        return fc2


def discriminator_conv28_batch(x, reuse=False):
    with tf.variable_scope('d_net') as vs:
        if reuse:
            vs.reuse_variables()
        conv1 = conv2d_lrelu(x, 64, 4, 2)
        conv2 = conv2d_lrelu(conv1, 128, 4, 2)
        conv2 = tf.reshape(conv2, [-1, np.prod(conv2.get_shape().as_list()[1:])])
        fc = fc_lrelu(conv2, 256)
        fc = tf.reshape(fc, [-1, 5120])
        fc = fc_lrelu(fc, 256)
        fc = tf.contrib.layers.fully_connected(fc, 1, activation_fn=tf.identity)
        return fc


discriminator_conv32 = discriminator_conv28
discriminator_conv32_batch = discriminator_conv28_batch


def discriminator_conv64(x, reuse=False):
    with tf.variable_scope('d_net') as vs:
        if reuse:
            vs.reuse_variables()
        conv = conv2d_lrelu(x, 64, 4, 2)
        conv = conv2d_lrelu(conv, 128, 4, 2)
        conv = conv2d_lrelu(conv, 192, 4, 2)
        conv = conv2d_lrelu(conv, 256, 4, 2)
        fc = tf.reshape(conv, [-1, np.prod(conv.get_shape().as_list()[1:])])
        fc = fc_lrelu(fc, 1024)
        fc = tf.contrib.layers.fully_connected(fc, 1, activation_fn=tf.identity)
        return fc


def discriminator_conv64large(x, reuse=False):
    with tf.variable_scope('d_net') as vs:
        if reuse:
            vs.reuse_variables()
        conv = conv2d_lrelu(x, 128, 4, 2)
        conv = conv2d_lrelu(conv, 256, 4, 2)
        conv = conv2d_lrelu(conv, 384, 4, 2)
        conv = conv2d_lrelu(conv, 512, 4, 2)
        fc = tf.reshape(conv, [-1, np.prod(conv.get_shape().as_list()[1:])])
        fc = fc_lrelu(fc, 2048)
        fc = tf.contrib.layers.fully_connected(fc, 1, activation_fn=tf.identity)
        return fc


def discriminator_conv64small(x, reuse=False):
    with tf.variable_scope('d_net') as vs:
        if reuse:
            vs.reuse_variables()
        conv = conv2d_lrelu(x, 48, 4, 2)
        conv = conv2d_lrelu(conv, 96, 4, 2)
        conv = conv2d_lrelu(conv, 128, 4, 2)
        conv = conv2d_lrelu(conv, 192, 4, 2)
        fc = tf.reshape(conv, [-1, np.prod(conv.get_shape().as_list()[1:])])
        fc = fc_lrelu(fc, 768)
        fc = tf.contrib.layers.fully_connected(fc, 1, activation_fn=tf.identity)
        return fc


def discriminator_fc64(x, reuse=False):
    with tf.variable_scope('d_net') as vs:
        if reuse:
            vs.reuse_variables()
        fc = tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])
        fc = fc_lrelu(fc, 1024)
        fc = fc_lrelu(fc, 1024)
        fc = fc_lrelu(fc, 1024)
        fc = fc_lrelu(fc, 1024)
        fc = tf.contrib.layers.fully_connected(fc, 1, activation_fn=tf.identity)
        return fc