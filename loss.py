import tensorflow as tf
import tensorflow.keras.backend as backend

def contrastive_loss(y, predictions, margin=1):
    y = tf.cast(y, predictions.dtype)

    squared_predicitons = backend.square(predictions)
    squared_margin = backend.square(backend.maximum(margin - predictions, 0))
    loss = backend.mean(y * squared_predicitons + (1 - y) * squared_margin)
    
    return loss