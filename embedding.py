from keras import Sequential
from keras.layers import Dense
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications import resnet
from config import target_image_shape


get_text_embedding = Sequential()
get_text_embedding.add(Dense(2048, input_shape=(370, ), activation='relu'))
get_text_embedding.add(Dense(512, activation='relu'))

base_cnn = resnet.ResNet50(
    weights="imagenet", input_shape=target_image_shape + (3,), include_top=False
)

flatten = layers.Flatten()(base_cnn.output)
dense1 = layers.Dense(1024, activation="relu")(flatten)
dense1 = layers.BatchNormalization()(dense1)
dense2 = layers.Dense(512, activation="relu")(dense1)
dense2 = layers.BatchNormalization()(dense2)
output = layers.Dense(512)(dense2)

get_image_embedding = Model(base_cnn.input, output, name="Embedding")

trainable = False
for layer in base_cnn.layers:
    if layer.name == "conv5_block1_out":
        trainable = True
    layer.trainable = trainable
