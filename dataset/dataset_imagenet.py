from .base import *


class ImagenetDataset(Dataset):
    def __init__(self, db_path='/data/data/imagenet/gen_train'):
        Dataset.__init__(self)
        self.data_dims = [64, 64, 3]
        self.name = "imagenet"

        self.db_path = db_path

        synset_file = open(os.path.join(self.db_path, "meta.txt"), "r")
        self.synset = []
        while True:
            line = synset_file.readline().split()
            if len(line) < 3:
                break
            self.synset.append([line[1], ' '.join(line[2:])])
        self.num_classes = len(self.synset)
        print("Total classes %d" % self.num_classes)

        # for ind, syn in enumerate(self.synset):
        #     if 'ocean' in syn[1]:
        #         print(ind, syn)

        self.db_files = os.listdir(self.db_path)
        self.db_files.remove("meta.txt")
        print(self.db_files)
        self.batch_ptr = 0

        self.cur_batch_ptr = -1
        self.cur_batch, self.cur_labels = self.load_new_data()
        self.train_data_ptr = 0
        self.train_size = len(self.db_files) * 10000
        self.test_size = self.train_size
        self.range = [-1.0, 1.0]

    def load_new_data(self):
        self.cur_batch_ptr += 1
        if self.cur_batch_ptr == len(self.db_files):
            self.cur_batch_ptr = 0
        filename = os.path.join(self.db_path, self.db_files[self.cur_batch_ptr])
        result = np.load(filename)
        return result['images'], result['labels']

    def next_batch(self, batch_size=None):
        return self.next_labeled_batch(batch_size)[0]

    def next_labeled_batch(self, batch_size):
        prev_data_ptr = self.train_data_ptr
        self.train_data_ptr += batch_size
        if self.train_data_ptr > self.cur_batch.shape[0]:
            self.train_data_ptr = batch_size
            prev_data_ptr = 0
            self.cur_batch, self.cur_labels = self.load_new_data()
        return self.cur_batch[prev_data_ptr:self.train_data_ptr, :, :, :], self.cur_labels[prev_data_ptr:self.train_data_ptr]

    def next_test_batch(self, batch_size=None):
        return self.next_batch(batch_size)

    def next_labeled_test_batch(self, batch_size=None):
        return self.next_labeled_batch(batch_size)

    def display(self, image):
        rescaled = np.divide(image + 1.0, 2.0)
        return np.clip(rescaled, 0.0, 1.0)

    def reset(self):
        self.batch_ptr = 0

    def label_name(self, index):
        return self.synset[index][1]

if __name__ == '__main__':
    dataset = ImagenetDataset()
    for j in range(100):
        for k in range(100):
            images, labels = dataset.next_labeled_batch()
        for i in range(0, 16):
            plt.subplot(4, 4, i+1)
            plt.imshow(dataset.display(images[i]))
            plt.gca().set_title(dataset.label_name(labels[i]))
        plt.show()