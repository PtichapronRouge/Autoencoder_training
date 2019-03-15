from keras.layers import Activation, Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras import backend as K


def build_encoder(image_size, latent_dim, layer_filters, kernel_size):
    input_shape = (image_size, image_size, 1)
    batch_size = 128
    kernel_size = 3

    inputs = Input(shape=input_shape, name='encoder_input')
    x = inputs

    for filters in layer_filters:
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=2,
                   activation='relu',
                   padding='same')(x)

    shape = K.int_shape(x)

    x = Flatten()(x)
    latent = Dense(latent_dim, name='latent_vector')(x)

    encoder = Model(inputs, latent, name='encoder')
    encoder.summary()

    return encoder, shape
