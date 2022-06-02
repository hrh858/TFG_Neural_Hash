import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('output/contrastive_siamese_model.h5', compile=False)

cut_model = tf.keras.Sequential()
cut_model.add(model.layers[0])
cut_model.add(model.layers[2])

# all_data = np.concatenate((np.load('output/data/train.npy'), np.load('output/data/test.npy')))
data = np.load('output/data/original.npy')

embeddings = []
for input in data:
    input = np.expand_dims(input, 0)
    embedding = cut_model.predict([input])
    embeddings.append(embedding)
np.save('output/embeddings/original', embeddings)

data = np.load('output/data/modified.npy')

embeddings = []
for input in data:
    tranform_emb = []
    for transf_im in input:
        transf_im = np.expand_dims(transf_im, 0)
        embedding = cut_model.predict([transf_im])
        tranform_emb.append(embedding)
    embeddings.append(tranform_emb)
np.save('output/embeddings/modified', embeddings)