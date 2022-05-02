import tensorboard
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import MaxPooling2D

def siamese_nn_model_basiccnn(in_shape, embedding_dim=128):
    inputs = Input(in_shape)


    y = Conv2D(64, (2,2), padding="same", activation="relu")(inputs)
    y = MaxPooling2D(pool_size=(2,2))(y)
    y = Dropout(0.3)(y)

    # y = Conv2D(64, (2,2), padding="same", activation="relu")(y)
    # y = MaxPooling2D(pool_size=(2,2))(y)
    # y = Dropout(0.3)(y)

    # y = Conv2D(64, (2,2), padding="same", activation="relu")(y)
    # y = MaxPooling2D(pool_size=(2,2))(y)
    # y = Dropout(0.3)(y)

    pooled_output = GlobalAveragePooling2D()(y)
    outputs = Dense(embedding_dim)(pooled_output)

    model = Model(inputs, outputs)

    return model

def siamese_nn_model_mobilenetv3():
    raise