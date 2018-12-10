from libgen.architecture import *


class WGAN:
    def __init__(self, dataset, log_path):
        self.log_path = log_path
        self.dataset = dataset

        if dataset.data_dims[0] == 28:
            discriminator = discriminator_conv28
            generator = generator_conv28
        else:
            discriminator = discriminator_conv64
            generator = generator_conv64

        self.z_dim = 50
        self.z = tf.placeholder(tf.float32, [None, self.z_dim])
        self.x = tf.placeholder(tf.float32, [None] + dataset.data_dims)

        self.g = generator(self.z)
        d = discriminator(self.x)
        d_ = discriminator(self.g, reuse=True)

        # Gradient penalty
        epsilon = tf.random_uniform([], 0.0, 1.0)
        x_hat = epsilon * self.x + (1 - epsilon) * self.g
        d_hat = discriminator(x_hat, reuse=True)

        ddx = tf.gradients(d_hat, x_hat)[0]
        ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=(1, 2, 3)))
        d_grad_loss = tf.reduce_mean(tf.square(ddx - 1.0) * 10.0)

        d_loss_x = -tf.reduce_mean(d)
        d_loss_g = tf.reduce_mean(d_)
        d_loss = d_loss_x + d_loss_g + d_grad_loss
        d_confusion = tf.reduce_mean(d) - tf.reduce_mean(d_)
        g_loss = -tf.reduce_mean(d_)

        d_vars = [var for var in tf.global_variables() if 'd_net' in var.name]
        g_vars = [var for var in tf.global_variables() if 'g_net' in var.name]

        self.d_train = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(d_loss, var_list=d_vars)
        self.g_train = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(g_loss, var_list=g_vars)

        self.train_summary = tf.summary.merge([
            tf.summary.scalar('g_loss', g_loss),
            tf.summary.scalar('d_loss', d_loss),
            tf.summary.scalar('confusion', d_confusion),
            tf.summary.scalar('d_loss_g', d_loss_g),
        ])

        self.sample_summary = tf.summary.merge([
            create_display(tf.reshape(self.g, [64]+dataset.data_dims), 'samples'),
            create_display(tf.reshape(self.x, [64]+dataset.data_dims), 'train_samples')
        ])

        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
        self.summary_writer = tf.summary.FileWriter(self.log_path)
        self.sess.run(tf.global_variables_initializer())

        self.idx = 1

    def save(self):
        saver = tf.train.Saver(
            var_list=[var for var in tf.global_variables() if 'd_net' in var.name or 'g_net' in var.name])
        saver.save(self.sess, os.path.join(self.log_path, "model.ckpt"))

    def train_step(self):
        batch_size = 100
        bx = self.dataset.next_batch(batch_size)
        bz = np.random.normal(size=(batch_size, self.z_dim))
        feed_dict = {self.x: bx, self.z: bz}
        self.sess.run([self.d_train, self.g_train], feed_dict=feed_dict)

        if self.idx % 100 == 0:
            summary_val = self.sess.run(self.train_summary, feed_dict=feed_dict)
            self.summary_writer.add_summary(summary_val, self.idx)

        if self.idx % 2000 == 0:
            summary_val = self.sess.run(self.sample_summary,
                                        feed_dict={self.x: bx[:64], self.z: np.random.normal(size=(64, self.z_dim))})
            self.summary_writer.add_summary(summary_val, self.idx)

        self.idx += 1