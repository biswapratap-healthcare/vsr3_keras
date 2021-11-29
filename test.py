import numpy as np
from keras.applications import resnet
from keras.models import load_model
from tensorflow import metrics
from data import preprocess_text, preprocess_image, train_dataset

count = 0
images = list()
texts = list()
while sample := next(iter(train_dataset)):
    image_anchor_input, text_positive_input, text_negative_input, \
    text_anchor_input, image_positive_input, image_negative_input = sample
    images.append(image_anchor_input)
    texts.append(text_positive_input)
    if count > 10:
        break
    count += 1


image_embedding_model = load_model('f2nn.h5')
text_embedding_model = load_model('t2nn.h5')

for i, image in enumerate(images):
    pp = resnet.preprocess_input(image)
    image_embedding = image_embedding_model(pp)
    result = dict()
    for j, text in enumerate(texts):
        text_embedding = text_embedding_model(text)
        cosine_similarity = metrics.CosineSimilarity()
        similarity = cosine_similarity(image_embedding, text_embedding)
        result[j] = similarity
    key_max = max(result, key=lambda x: result[x])
    print(i, key_max)
