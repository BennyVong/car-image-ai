import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet_v2 import preprocess_input


def get_label(filename: str, label_type: str = 'make'):
    parts = tf.strings.split(filename, '/')
    name = parts[-1]
    label = tf.strings.split(name, '_')
    if label_type == 'make':
        return tf.strings.lower(label[0])
    elif label_type == 'makemodel':
        return tf.strings.lower(label[0] + '_' + label[1])
    elif label_type == 'makemodelyear':
        return tf.strings.lower(label[0] + '_' + label[1] + '_' + label[2])
    else:
        raise ValueError('label must be either "make" or "makemodel" or "makemodelyear" and not ', label_type)

def get_image(filename: str, size: tuple = (212, 320)):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize_with_crop_or_pad(image, target_height=size[0], target_width=size[1])
    image = preprocess_input(image)
    return image

def one_hot_encode(classes: list, label):
    n_classes = len(classes)
    names = tf.constant(classes, shape=[1, n_classes])
    index = tf.argmax(tf.cast(tf.equal(names, label), tf.int32), axis=1)
    y = tf.one_hot(index, n_classes)
    y = tf.squeeze(y)
    return y

def parse_file(filename: str, classes: list, input_size: tuple, label_type: str):
    label = get_label(filename, label_type=label_type)
    image = get_image(filename, size=input_size)

    target = one_hot_encode(classes=classes, label=label)

    return image, target

def image_augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    return image, label

def construct_ds(input_files: list, batch_size: int, classes: list, label_type: str, input_size: tuple = (212, 320), prefetch_size: int = 10, shuffle_size: int = 32, shuffle: bool = True, augment: bool = False):
    ds = tf.data.Dataset.from_tensor_slices(input_files)

    if shuffle:
        ds = ds.shuffle(buffer_size=shuffle_size)

    ds = ds.map(lambda x: parse_file(x, classes=classes, input_size=input_size, label_type=label_type))

    if augment and tf.random.uniform((), minval=0, maxval=1, dtype=tf.dtypes.float32, seed=None, name=None) < 0.7:
        ds = ds.map(image_augment)

    ds = ds.batch(batch_size=batch_size)
    ds = ds.prefetch(buffer_size=prefetch_size)

    return ds

def show_batch(ds: tf.data.Dataset, classes: list, rescale: bool = False, size: tuple = (10, 10), title: str = None):
    plt.figure(figsize=size)

    for image, label in ds.take(1):
        image_array = image.numpy()
        image_array += 1.0
        image_array /= 2.0
        label_array = label.numpy()
        batch_size = image_array.shape[0]
        for idx in range(batch_size):
            label = classes[np.argmax(label_array[idx])]
            if rescale:
                plt.imshow(image_array[idx] * 255)
            else:
                plt.imshow(image_array[idx])
            plt.title(label + ' ' + str(image_array[idx].shape), fontsize=10)
            plt.axis('off')

    if title is not None:
        plt.suptitle(title)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
