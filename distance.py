import tensorflow.keras.backend as K

def euclidean_distance(vecs):
    vec_a, vec_b = vecs
    squared_sum = K.sum(K.square(vec_a-vec_b), axis=1, keepdims=True)
    return K.sqrt(K.maximum(squared_sum, K.epsilon()))