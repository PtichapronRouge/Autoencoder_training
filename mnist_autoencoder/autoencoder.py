from keras.layers import Activation, Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
import numpy as np

from encoder import build_encoder
from decoder import build_decoder

(X_train, _), (X_test, _) = mnist.load_data()

image_size = X_train.shape[1]
X_train = np.reshape(X_train, [-1, image_size, image_size, 1])
X_test = np.reshape(X_test, [-1, image_size, image_size, 1])
X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255

latent_dim= 16
batch_size = 128
kernel_size = 3
layer_filters = [32,64]

inputs, encoder, shape = build_encoder(image_size, latent_dim, layer_filters, kernel_size)
decoder = build_decoder(shape, latent_dim, layer_filters, kernel_size)

autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
autoencoder.summary()
autoencoder.compile(loss='mse', optimizer='adam')

autoencoder.fit(X_train,
                X_train,
                validation_data=(X_test, X_test),
                epochs=30,
                batch_size=batch_size)

