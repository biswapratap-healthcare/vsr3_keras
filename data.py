import os
import numpy as np
import pandas as pd
import tensorflow as tf
from one_hot_map import one_hot_map
from config import target_image_shape, plate_images_path, plate_text_path


def preprocess_text(txt):
    txt = txt.numpy().decode("utf-8")
    length = len(txt)
    if length < 10:
        num = 10 - length
        for _ in range(0, num, 1):
            txt = txt + '#'
    elif length > 10:
        txt = txt[0:10]
    encoding = list()
    for t in txt:
        encoding.extend(one_hot_map.get(t))
    encoding = np.array(encoding)
    encoding = tf.convert_to_tensor(encoding, dtype=tf.float32)
    return encoding


def preprocess_image(filename):
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, target_image_shape)
    return image


def preprocess_symmetric_triplets(anchor_image, positive_text, negative_text,
                                  anchor_text, positive_image, negative_image):
    tf_positive_text = tf.py_function(preprocess_text, [positive_text], tf.float32)
    tf_negative_text = tf.py_function(preprocess_text, [negative_text], tf.float32)
    tf_anchor_text = tf.py_function(preprocess_text, [anchor_text], tf.float32)
    return (
               preprocess_image(anchor_image),
               tf_positive_text,
               tf_negative_text,
               tf_anchor_text,
               preprocess_image(positive_image),
               preprocess_image(negative_image)
    )


plate_images = list()
for f in os.listdir(plate_images_path):
    if 'troi' not in f:
        plate_images.append(plate_images_path + '/' + f)
plate_images = sorted(plate_images)[:200]


plate_texts = list()
df = pd.read_csv(plate_text_path)
df_dict = df.set_index('imgID').to_dict()
for idx, plate_image in enumerate(plate_images):
    imgID = os.path.basename(plate_image).split('.')[0]
    plate_text = df_dict.get('GT').get(imgID)
    if plate_text is not None:
        plate_texts.append(plate_text)
    else:
        plate_images.pop(idx)

plate_count = len(plate_images)

image_anchor_dataset = tf.data.Dataset.from_tensor_slices(plate_images)
text_positive_dataset = tf.data.Dataset.from_tensor_slices(plate_texts)

text_anchor_dataset = tf.data.Dataset.from_tensor_slices(plate_texts)
image_positive_dataset = tf.data.Dataset.from_tensor_slices(plate_images)

rng = np.random.RandomState(seed=42)
rng.shuffle(plate_texts)

text_negative_dataset = tf.data.Dataset.from_tensor_slices(plate_texts)

rng = np.random.RandomState(seed=24)
rng.shuffle(plate_images)
image_negative_dataset = tf.data.Dataset.from_tensor_slices(plate_images)

dataset = tf.data.Dataset.zip((image_anchor_dataset, text_positive_dataset, text_negative_dataset,
                               text_anchor_dataset, image_positive_dataset, image_negative_dataset))

dataset = dataset.shuffle(buffer_size=4096)
dataset = dataset.map(preprocess_symmetric_triplets)

# Let's now split our dataset in train and validation.
train_dataset = dataset.take(round(plate_count * 0.8))
val_dataset = dataset.skip(round(plate_count * 0.8))

train_dataset = train_dataset.batch(32, drop_remainder=False)
train_dataset = train_dataset.prefetch(8)

val_dataset = val_dataset.batch(32, drop_remainder=False)
val_dataset = val_dataset.prefetch(8)
