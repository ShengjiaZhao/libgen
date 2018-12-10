from .layers import *


# Input a tensor of shape [batch_size, height, width, color_dim], tiles the tensor for display
# color_dim can be 1 or 3. The tensor should be normalized to lie in [0, 1]
def create_display(tensor, name):
    channels, height, width, color_dim = [tensor.get_shape()[i].value for i in range(4)]
    cnt = int(math.floor(math.sqrt(float(channels))))
    if color_dim >= 3:
        tensor = tf.slice(tensor, [0, 0, 0, 0], [cnt * cnt, -1, -1, 3])
        color_dim = 3
    else:
        tensor = tf.slice(tensor, [0, 0, 0, 0], [cnt * cnt, -1, -1, 1])
        color_dim = 1
    tensor = tf.pad(tensor, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
    tensor = tf.transpose(tensor, perm=[1, 0, 2, 3])
    tensor = tf.reshape(tensor, [height + 2, cnt, cnt, width + 2, color_dim])
    tensor = tf.transpose(tensor, perm=[1, 0, 2, 3, 4])
    tensor = tf.reshape(tensor, [1, (height + 2) * cnt, (width + 2) * cnt, color_dim])
    return tf.summary.image(name, tensor, max_outputs=1)


def create_multi_display(tensors, name):
    channels, height, width, color_dim = [tensors[0].get_shape()[i].value for i in range(4)]
    max_columns = 15
    columns = int(math.floor(float(max_columns) / len(tensors)))
    rows = int(math.floor(float(channels) / columns))
    if rows == 0:
        columns = channels
        rows = 1

    for index in range(len(tensors)):
        if color_dim >= 3:
            tensors[index] = tf.slice(tensors[index], [0, 0, 0, 0], [rows * columns, -1, -1, 3])
            color_dim = 3
        else:
            tensors[index] = tf.slice(tensors[index], [0, 0, 0, 0], [rows * columns, -1, -1, 1])
            color_dim = 1
    tensor = tf.stack(tensors)
    tensor = tf.transpose(tensor, [1, 0, 2, 3, 4])
    tensor = tf.reshape(tensor, [-1, height, width, color_dim])
    tensor = tf.pad(tensor, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
    tensor = tf.transpose(tensor, perm=[1, 0, 2, 3])
    tensor = tf.reshape(tensor, [height + 2, rows, columns * len(tensors), width + 2, color_dim])
    tensor = tf.transpose(tensor, perm=[1, 0, 2, 3, 4])
    tensor = tf.reshape(tensor, [1, (height + 2) * rows, (width + 2) * columns * len(tensors), color_dim])
    return tf.summary.image(name, tensor, max_outputs=1)


