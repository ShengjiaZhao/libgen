from .base import *


class MoGDataset(Dataset):
    def __init__(self, num_gaussian=10):
        Dataset.__init__(self)
        self.input_dim = 1
        self.batch_size = 1000
        self.name = "mog"
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.05)
        self.sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
        self.init_distribution(num_gaussian)

    def next_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        centers = np.random.choice(self.distribution_centers, size=batch_size, replace=True, p=self.distribution_weights)
        return np.reshape(centers, (batch_size, 1)) + np.random.normal(size=(batch_size, 1), scale=self.sample_noise)

    def ground_truth(self):
        x_range = np.array(np.linspace(-0.1, 1.1, num=2000))
        return x_range, self.density(x_range)

    """ This computes the gradient of log likelihood in a numerically stable way
    def density_gradient(self, x_range):
        x_range = x_range.flatten()
        num_mixture = len(self.distribution_centers)
        weighted_ll = np.ndarray((x_range.size, num_mixture), np.float)
        grad = np.ndarray((x_range.size, num_mixture), np.float)
        for index, (center, weight) in enumerate(zip(self.distribution_centers, self.distribution_weights)):
            weighted_ll[:, index] = -np.divide(np.square(np.subtract(x_range, center)),
                                               2 * self.sample_noise ** 2) + np.log(weight)
            grad[:, index] = -np.divide(x_range - center, 2 * self.sample_noise ** 2)
        e_x = np.exp(weighted_ll - np.tile(np.max(weighted_ll, axis=1, keepdims=True), (1, num_mixture)))
        softmax_weight = e_x / np.tile(np.sum(e_x, axis=1, keepdims=True), (1, num_mixture))
        return np.sum(np.multiply(softmax_weight, grad), axis=1)
    """

    """ Take as input x_input a tensor of shape (batch_size, 1), and return gradients of size (batch_size, 1) """
    def build_density_gradient(self, x_input):
        num_mixture = len(self.distribution_centers)
        expanded_input = tf.tile(x_input, multiples=(1, num_mixture))
        weighted_ll = -tf.div(tf.square(tf.sub(expanded_input, self.tf_distribution_centers)),
                             2 * self.sample_noise ** 2) + tf.log(self.tf_distribution_weights)
        grad = -np.divide(expanded_input - self.tf_distribution_centers, 2 * self.sample_noise ** 2)
        return tf.reduce_sum(tf.mul(tf.nn.softmax(weighted_ll), grad), reduction_indices=-1, keep_dims=True)

    """ x_range is a tensor of shape (batch_size, 1), returns the density of shape (batch_size, 1) evaluated at these x """
    def build_density(self, x_input):
        expanded_input = tf.tile(x_input, multiples=(1, len(self.distribution_centers)))
        tf_density = tf.exp(-tf.div(tf.square(expanded_input - self.tf_distribution_centers),
                                    2 * self.sample_noise ** 2))
        tf_density = 1 / math.sqrt(2 * math.pi) / self.sample_noise * \
            tf.reduce_sum(tf.mul(tf_density, self.tf_distribution_weights), reduction_indices=1, keep_dims=True)
        return tf_density

    """
    def density_gradient(self, x_range):
        y_val = np.zeros(x_range.shape)
        nll = np.zeros()
        for index, (center, weight) in enumerate(zip(self.distribution_centers, self.distribution_weights)):
            nll = np.divide(np.square(np.subtract(x_range, center)), 2 * self.sample_noise ** 2)
            y_val += -weight / math.sqrt(2 * math.pi) / self.sample_noise * \
                np.exp(-nll) * \
                np.divide(x_range - center, self.sample_noise ** 2)
        y_val = np.divide(y_val, self.density(x_range))
        return y_val
    """
    def density(self, x_range):
        # Compute the distribution curve
        # y_val = np.zeros(x_range.shape)
        return self.sess.run(self.density_op, feed_dict={self.input_placeholder: np.reshape(x_range, (x_range.size, 1))})

        # for center, weight in zip(self.distribution_centers, self.distribution_weights):
        #     y_val += weight / math.sqrt(2 * math.pi) / self.sample_noise * \
        #              np.exp(-np.divide(np.square(np.subtract(x_range, center)), 2 * self.sample_noise**2))
        # return y_val

    def density_gradient(self, x_range):
        x_range = np.reshape(x_range, (x_range.size, 1))
        gradient = self.sess.run(self.density_gradient_op, feed_dict={self.input_placeholder: x_range})
        return gradient.flatten()

    def init_distribution(self, num_gaussian):
        np.random.seed(1024)
        initial_centers = int(math.ceil(math.sqrt(num_gaussian)))
        secondary_centers = int(num_gaussian / initial_centers)
        interval = 1.0 / initial_centers / 2.0

        centers = np.linspace(0.0, 1.0, num=initial_centers)
        centers_weights = np.random.random(initial_centers)
        centers_weights /= np.sum(centers_weights)

        secondaries = np.zeros(initial_centers*secondary_centers)
        secondary_weights = np.zeros(initial_centers*secondary_centers)
        for i in range(0, initial_centers):
            secondaries[i*secondary_centers:(i+1)*secondary_centers] = centers[i] + interval * np.random.random(secondary_centers)
            rand_weights = np.random.random(secondary_centers)
            secondary_weights[i*secondary_centers:(i+1)*secondary_centers] = centers_weights[i] * (rand_weights / sum(rand_weights))
        assert abs(np.sum(secondary_weights) - 1) < 0.001

        self.distribution_centers = secondaries
        self.distribution_weights = secondary_weights
        self.sample_noise = 1.0 / num_gaussian / 4.0

        with tf.name_scope("distribution_density"):
            self.tf_distribution_centers = tf.constant(self.distribution_centers, tf.float32)
            self.tf_distribution_weights = tf.constant(self.distribution_weights, tf.float32)
            self.input_placeholder = tf.placeholder(tf.float32, shape=(None, 1))
            self.density_op = self.build_density(self.input_placeholder)
            self.density_gradient_op = self.build_density_gradient(self.input_placeholder)
