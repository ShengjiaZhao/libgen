from libgen.architecture import *


class VAE:
    def __init__(self, dataset, log_path, binary=False):
        self.dataset = dataset
        self.log_path = log_path

        if dataset.data_dims[0] == 28:
            encoder = encoder_conv28
            generator = generator_conv28
        else:
            encoder = encoder_conv64
            generator = generator_conv64

        # Build the computation graph for training
        self.z_dim = 20
        self.train_x = tf.placeholder(tf.float32, shape=[None] + dataset.data_dims)
        train_zdist, train_zsample = encoder(self.train_x, self.z_dim)
        # ELBO loss divided by input dimensions
        zkl_per_sample = tf.reduce_sum(-tf.log(train_zdist[1]) + 0.5 * tf.square(train_zdist[1]) +
                                       0.5 * tf.square(train_zdist[0]) - 0.5, axis=1)
        loss_zkl = tf.reduce_mean(zkl_per_sample)
        train_xr = generator(train_zsample)

        # Build the computation graph for generating samples
        self.gen_z = tf.placeholder(tf.float32, shape=[None, self.z_dim])
        self.gen_x = generator(self.gen_z, reuse=True)

        # Negative log likelihood per dimension
        if binary:
            nll_per_sample = -tf.reduce_sum(tf.log(train_xr) * self.train_x + tf.log(1 - train_xr) * (1 - self.train_x),
                                                 axis=(1, 2, 3))
        else:
            nll_per_sample = tf.reduce_sum(tf.square(self.train_x - train_xr) + 0.5 * tf.abs(self.train_x - train_xr), axis=(1, 2, 3))

        loss_nll = tf.reduce_mean(nll_per_sample)

        self.kl_anneal = tf.placeholder(tf.float32)
        loss_elbo = loss_nll + loss_zkl * self.kl_anneal
        self.trainer = tf.train.AdamOptimizer(1e-4, beta1=0.5, beta2=0.9).minimize(loss_elbo)

        self.train_summary = tf.summary.merge([
            tf.summary.scalar('loss_zkl', loss_zkl),
            tf.summary.scalar('loss_nll', loss_nll),
            tf.summary.scalar('loss_elbo', loss_elbo),
        ])

        self.sample_summary = tf.summary.merge([
            create_display(tf.reshape(self.gen_x, [64]+dataset.data_dims), 'samples'),
            create_display(tf.reshape(train_xr, [64]+dataset.data_dims), 'reconstructions'),
            create_display(tf.reshape(self.train_x, [64]+dataset.data_dims), 'train_samples')
        ])

        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
        self.summary_writer = tf.summary.FileWriter(log_path)
        self.sess.run(tf.global_variables_initializer())

        self.idx = 1

    def train_step(self):
        batch_size = 100
        bx = self.dataset.next_batch(batch_size)
        self.sess.run(self.trainer, feed_dict={self.train_x: bx, self.kl_anneal: 1 - math.exp(-self.idx / 20000.0)})

        if self.idx % 100 == 0:
            summary_val = self.sess.run(self.train_summary,
                                        feed_dict={self.train_x: bx, self.kl_anneal: 1 - math.exp(-self.idx / 20000.0)})
            self.summary_writer.add_summary(summary_val, self.idx)

        if self.idx % 2000 == 0:
            summary_val = self.sess.run(self.sample_summary,
                                        feed_dict={self.train_x: bx[:64],
                                                   self.gen_z: np.random.normal(size=(64, self.z_dim))})
            self.summary_writer.add_summary(summary_val, self.idx)

        self.idx += 1

    def save(self):
        saver = tf.train.Saver(
            var_list=[var for var in tf.global_variables() if 'i_net' in var.name or 'g_net' in var.name])
        saver.save(self.sess, os.path.join(self.log_path, "model.ckpt"))