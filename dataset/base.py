import numpy as np
import math
from matplotlib import pyplot as plt
import tensorflow as tf
import os
from glob import glob
from scipy import misc


class Dataset:
    def __init__(self):
        self.batch_size = -1
        self.name = "abstract"
        self.data_dims = []
        self.width = -1
        self.height = -1
        self.train_size = -1
        self.test_size = -1
        self.range = [0.0, 1.0]

    """ Get next training batch """
    def next_batch(self, batch_size):
        self.handle_unsupported_op()
        return None

    def next_test_batch(self, batch_size):
        self.handle_unsupported_op()
        return None

    def display(self, image):
        return image

    """ After reset, the same batches are output with the same calling order of next_batch or next_test_batch"""
    def reset(self):
        self.handle_unsupported_op()

    def handle_unsupported_op(self):
        print("Unsupported Operation")
        raise(Exception("Unsupported Operation"))

    def full_train_batch(self):
        self.handle_unsupported_op()

    def full_test_batch(self):
        self.handle_unsupported_op()