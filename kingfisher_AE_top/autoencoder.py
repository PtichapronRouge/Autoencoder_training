from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
import numpy as np
import sys

data_path = "/tmp/data.npy"
if len(sys.argv) > 1:
    data_path = str(sys.argv[1])

print("Loading data...")
data = np.load(data_path)
#shape = (2451, 512, 512)
image_size = data.shape[1]
input_shape=(image_size,image_size,1)

print("Generating layers...")
data = np.reshape(data, [-1, image_size, image_size, 1])
data = data.astype('float32')/255

input_img = Input(shape=input_shape)

x = Conv2D(16, (5,5), activation='relu', padding='same')(input_img) #512*512
x = MaxPooling2D((4,4), padding='same')(x)                          #128*128
x = Conv2D(8, (5,5), activation='relu', padding='same')(x)          #128*128
x = MaxPooling2D((4,4), padding='same')(x)                    #32*32
x = Conv2D(8, (5,5), activation='relu', padding='same')(x)          #128*128
encoded = MaxPooling2D((4,4), padding='same')(x)                    #32*32

x = Conv2D(8, (5,5), activation='relu', padding='same')(encoded)
x = UpSampling2D((4,4))(x)
x = Conv2D(8, (5,5), activation='relu', padding='same')(x)
x = UpSampling2D((4,4))(x)
x = Conv2D(16, (5,5), activation='relu', padding='same')(x)
x = UpSampling2D((4,4))(x)
decoded = Conv2D(1, (5,5), activation='relu', padding='same')(x)

autoencoder = Model(input_img, decoded)
print("Compiling Model")
autoencoder.compile(optimizer='adadelta',
                    loss='binary_crossentropy',
                    metrics=['loss', 'accuracy'])

print("Training...")
autoencoder.fit(data, data,
                epochs=20,
                batch_size=64,
                shuffle=True,
                validation_split=0.1)
