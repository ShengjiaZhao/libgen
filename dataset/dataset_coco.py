from .base import *
import pickle

class CocoDataset(Dataset):
    def __init__(self):
        Dataset.__init__(self)
        self.batch_size = 100
        self.data_dims = [64, 64]
        self.name = "coco"

        self.db_path = "coco/images"
        self.label_path = "coco/label.p"
        self.categories = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)), self.label_path), "rb"))

        self.db_files = os.listdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), self.db_path))
        self.test_db_file = self.db_files[0]
        self.train_db_files = self.db_files[1:]

        self.cur_batch_ptr = 0
        self.cur_batch_img, self.cur_batch_mask = self.load_new_data()
        self.train_batch_ptr = 0
        self.train_size = len(self.train_db_files) * 10000

        test_data = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                         self.db_path, self.test_db_file))
        self.test_img, self.test_mask = test_data['img'], test_data['mask']
        self.test_batch_ptr = 0
        self.test_size = self.train_size

        self.num_classes = len(self.categories) + 1  # Add 1 for background class
        self.color_map = np.random.rand(255, 3)

    def load_new_data(self):
        filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                self.db_path, self.db_files[self.cur_batch_ptr])
        self.cur_batch_ptr += 1
        if self.cur_batch_ptr == len(self.db_files):
            self.cur_batch_ptr = 0
        data = np.load(filename)
        return data['img'], data['mask']

    def next_batch_mask(self, batch_size=None):
        return self.next_batch_with_mask(batch_size)[1]

    def next_batch(self, batch_size=None):
        return self.next_batch_with_mask(batch_size)[0]

    def next_batch_with_mask(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        prev_batch_ptr = self.train_batch_ptr
        self.train_batch_ptr += batch_size
        if self.train_batch_ptr > self.cur_batch_img.shape[0]:
            self.train_batch_ptr = batch_size
            prev_batch_ptr = 0
            self.cur_batch_img, self.cur_batch_mask = self.load_new_data()
        return self.cur_batch_img[prev_batch_ptr:self.train_batch_ptr, :, :, :], \
               self.cur_batch_mask[prev_batch_ptr:self.train_batch_ptr, :, :]

    def next_test_batch_mask(self, batch_size=None):
        return self.next_test_batch_with_mask(batch_size)[1]

    def next_test_batch(self, batch_size=None):
        return self.next_test_batch_with_mask(batch_size)[0]

    def next_test_batch_with_mask(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        prev_batch_ptr = self.test_batch_ptr
        self.test_batch_ptr += batch_size
        if self.test_batch_ptr > self.test_img.shape[0]:
            self.test_batch_ptr = batch_size
            prev_batch_ptr = 0
        return self.test_img[prev_batch_ptr:self.test_batch_ptr, :, :, :], \
               self.test_mask[prev_batch_ptr:self.test_batch_ptr, :, :]

    def display(self, image, mask=None):
        return self.display_image(image, mask)

    """ Transform image to displayable """
    def display_image(self, image, mask=None):
        if mask is None:
            return np.clip(image, 0.0, 1.0)
        else:
            return np.clip(image, 0.0, 1.0) * 0.8 + self.display_mask(mask) * 0.2

    def display_mask(self, mask):
        # canvas = np.zeros((mask.shape[0] * 2, mask.shape[1] * 2, 3), dtype=np.float32)

        colored = self.color_map[mask]
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.imshow(colored)
        # ax.text(x=0.5, y=0.6, s="Hello World")
        # fig.canvas.draw()
        # data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        # data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return colored

if __name__ == '__main__':
    dataset = CocoDataset()
    while True:
        batch, mask = dataset.next_batch_with_mask()
        plt.subplot(1, 2, 1)
        plt.imshow(dataset.display(batch[0]))
        plt.subplot(1, 2, 2)
        plt.imshow(dataset.display_mask(mask[0]))
        plt.show()
        for i in range(49):
            dataset.next_batch()
