import tensorflow as tf
import tensorflow.keras.backend as K

def contrastive_loss(y, predictions, margin=1):
    y = tf.cast(y, predictions.dtype)

    squared_predicitons = K.square(predictions)
    squared_margin = K.square(K.maximum(margin - predictions, 0))
    loss = K.mean(y * squared_predicitons + (1 - y) * squared_margin)
    return loss