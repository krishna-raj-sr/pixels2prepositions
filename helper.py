import tensorflow as tf
import constants as c
import numpy as np

def make_dataset(images, captions, batch_size):
    img_dataset = tf.data.Dataset.from_tensor_slices(images).map(read_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    cap_dataset = tf.data.Dataset.from_tensor_slices(np.stack(captions.values,0))
    dataset = tf.data.Dataset.zip((img_dataset, cap_dataset))
    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def read_image(img_path,size = c.IMAGE_SIZE, colormode = c.COLOR_MODE):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, size)
    img = tf.image.convert_image_dtype(img, tf.float32)
    if colormode == "HSV":
      img = tf.image.rgb_to_hsv(img)
    if colormode == 'YUV':
      img = tf.image.rgb_to_yuv(img)
    return img




def split_data(dataset, train_ratio=c.TRAIN_VALIDATION_TEST_RATIO[0], valid_ratio=c.TRAIN_VALIDATION_TEST_RATIO[1], test_ratio=c.TRAIN_VALIDATION_TEST_RATIO[2], random_state=166):
    # Shuffle the dataset
    shuffled_data = dataset.sample(frac=1, random_state=random_state)
    
    # Calculate lengths for each split
    train_len = int(train_ratio * len(shuffled_data))
    valid_len = int(valid_ratio * len(shuffled_data))
    
    # Split the dataset
    train_data = shuffled_data[:train_len]
    valid_data = shuffled_data[train_len:train_len+valid_len]
    test_data = shuffled_data[train_len+valid_len:]
    
    return train_data, valid_data, test_data

# Usage:


