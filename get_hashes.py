import numpy as np

original_embeddings = np.load('output/embeddings/original.npy')
modified_embeddings = np.load('output/embeddings/modified.npy')

or_binary_embs = []
for embedding in original_embeddings:
    aux = np.where(embedding < 0, 0, embedding)
    aux = np.where(aux > 0, 1, aux)
    or_binary_embs.append(aux)
or_binary_embs = np.array(or_binary_embs)

transf_binary_embeddings = []
for embedding_group in modified_embeddings:
    bin_embs = []
    for trnasf_em in embedding_group:
        aux = np.where(trnasf_em < 0, 0, trnasf_em)
        aux = np.where(aux > 0, 1, aux)
        bin_embs.append(aux)
    transf_binary_embeddings.append(bin_embs)
transf_binary_embeddings = np.array(transf_binary_embeddings)

np.save('output/hashes/original', or_binary_embs)
np.save('output/hashes/modified', transf_binary_embeddings)