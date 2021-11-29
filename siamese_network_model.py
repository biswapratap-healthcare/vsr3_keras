import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications import resnet

from config import target_image_shape, target_text_shape
from embedding import get_image_embedding, get_text_embedding


class DistanceLayer(layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, image_anchor, text_positive, text_negative,
             text_anchor, image_positive, image_negative):
        itt_ap_distance = tf.reduce_sum(tf.square(image_anchor - text_positive), -1)
        itt_an_distance = tf.reduce_sum(tf.square(image_anchor - text_negative), -1)
        tii_ap_distance = tf.reduce_sum(tf.square(text_anchor - image_positive), -1)
        tii_an_distance = tf.reduce_sum(tf.square(text_anchor - image_negative), -1)
        return itt_ap_distance, itt_an_distance, tii_ap_distance, tii_an_distance


image_anchor_input = layers.Input(name="image_anchor", shape=target_image_shape + (3,))
text_positive_input = layers.Input(name="text_positive", shape=target_text_shape)
text_negative_input = layers.Input(name="text_negative", shape=target_text_shape)

text_anchor_input = layers.Input(name="text_anchor", shape=target_text_shape)
image_positive_input = layers.Input(name="image_positive", shape=target_image_shape + (3,))
image_negative_input = layers.Input(name="image_negative", shape=target_image_shape + (3,))

distances = DistanceLayer()(
    get_image_embedding(resnet.preprocess_input(image_anchor_input)),
    get_text_embedding(text_positive_input),
    get_text_embedding(text_negative_input),
    get_text_embedding(text_anchor_input),
    get_image_embedding(resnet.preprocess_input(image_positive_input)),
    get_image_embedding(resnet.preprocess_input(image_negative_input)),
)

siamese_network = Model(
    inputs=[image_anchor_input, text_positive_input, text_negative_input,
            text_anchor_input, image_positive_input, image_negative_input], outputs=distances
)
