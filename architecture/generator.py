from .layers import *


def generator_conv64(z, x_dim=3, reuse=False):
    with tf.variable_scope('g_net') as vs:
        if reuse:
            vs.reuse_variables()
        fc = fc_bn_relu(z, 1024)
        fc = fc_bn_relu(fc, 4*4*256)
        conv = tf.reshape(fc, tf.stack([tf.shape(fc)[0], 4, 4, 256]))
        conv = conv2d_t_bn_relu(conv, 192, 4, 2)
        conv = conv2d_t_relu(conv, 128, 4, 2)
        conv = conv2d_t_relu(conv, 128, 4, 1)
        conv = conv2d_t_relu(conv, 64, 4, 2)
        output = tf.contrib.layers.convolution2d_transpose(conv, x_dim, 4, 2, activation_fn=tf.sigmoid)
        return output


def generator_conv64small(z, x_dim=3, reuse=False):
    with tf.variable_scope('g_net') as vs:
        if reuse:
            vs.reuse_variables()
        fc = fc_relu(z, 768)
        fc = fc_relu(fc, 4*4*192)
        conv = tf.reshape(fc, tf.stack([tf.shape(fc)[0], 4, 4, 192]))
        conv = conv2d_t_relu(conv, 128, 4, 2)
        conv = conv2d_t_relu(conv, 96, 4, 2)
        conv = conv2d_t_relu(conv, 96, 4, 1)
        conv = conv2d_t_relu(conv, 48, 4, 2)
        output = tf.contrib.layers.convolution2d_transpose(conv, x_dim, 4, 2, activation_fn=tf.sigmoid)
        return output


def generator_conv64large(z, x_dim=3, reuse=False):
    with tf.variable_scope('g_net') as vs:
        if reuse:
            vs.reuse_variables()
        fc = fc_relu(z, 2048)
        fc = fc_relu(fc, 4*4*512)
        conv = tf.reshape(fc, tf.stack([tf.shape(fc)[0], 4, 4, 512]))
        conv = conv2d_t_relu(conv, 384, 4, 2)
        conv = conv2d_t_relu(conv, 256, 4, 2)
        conv = conv2d_t_relu(conv, 256, 4, 1)
        conv = conv2d_t_relu(conv, 128, 4, 2)
        output = tf.contrib.layers.convolution2d_transpose(conv, x_dim, 4, 2, activation_fn=tf.sigmoid)
        return output


def generator_fc64(z, x_dim=3, reuse=False):
    with tf.variable_scope('g_net') as vs:
        if reuse:
            vs.reuse_variables()
        fc = fc_bn_relu(z, 1024)
        fc = fc_bn_relu(fc, 1024)
        fc = fc_bn_relu(fc, 1024)
        fc = fc_bn_relu(fc, 1024)
        fc = tf.contrib.layers.fully_connected(fc, 64*64*3, activation_fn=tf.sigmoid)
        output = tf.reshape(fc, tf.stack([tf.shape(fc)[0], 64, 64, x_dim]))
        return output


def generator_conv28(z, x_dim=1, reuse=False):
    with tf.variable_scope('g_net') as vs:
        if reuse:
            vs.reuse_variables()
        fc = fc_relu(z, 1024)
        fc = fc_relu(fc, 7*7*128)
        fc = tf.reshape(fc, tf.stack([tf.shape(fc)[0], 7, 7, 128]))
        conv = conv2d_t_relu(fc, 64, 4, 2)
        conv = conv2d_t_relu(conv, 64, 4, 1)
        output = tf.contrib.layers.convolution2d_transpose(conv, x_dim, 4, 2, activation_fn=tf.sigmoid)
        return output


def generator_conv32(z, x_dim=3, reuse=False):
    with tf.variable_scope('g_net') as vs:
        if reuse:
            vs.reuse_variables()
        fc = fc_relu(z, 1024)
        fc = fc_relu(fc, 8*8*128)
        fc = tf.reshape(fc, tf.stack([tf.shape(fc)[0], 8, 8, 128]))
        conv = conv2d_t_relu(fc, 64, 4, 2)
        conv = conv2d_t_relu(conv, 64, 4, 1)
        output = tf.contrib.layers.convolution2d_transpose(conv, x_dim, 4, 2, activation_fn=tf.sigmoid)
        return output
