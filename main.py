from tensorflow.keras import optimizers
from data import train_dataset, val_dataset
from siamese_model import SiameseModel
from siamese_network_model import siamese_network
from embedding import get_text_embedding, get_image_embedding

siamese_model = SiameseModel(siamese_network)
siamese_model.compile(optimizer=optimizers.Adam(0.0001))
siamese_model.fit(train_dataset, epochs=30, validation_data=val_dataset)

get_text_embedding.save('t2nn.h5')
get_image_embedding.save('f2nn.h5')
