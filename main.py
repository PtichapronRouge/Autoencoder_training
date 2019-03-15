from keras.datasets import mnist
from keras.utils import to_categorical
from keras.layers import Input, Dense, Activation, Lambda
from keras.models import Model
from keras.callbacks import TensorBoard

import sys
import os

run_name = "linear"
log_path = generate_unique_logpath("./logs_linear", run_name)
tbcb = TensorBoard(log_dir=log_path)

def generate_unique_logpath(logdir, raw_run_name):
    i = 0
    while(True):
        run_name = raw_run_name + "-" + str(i)
        log_path = os.path.join(logdir, run_name)
        if not os.path.isdir(log_path):
            return log_path
        i++

act='softmax'
if len(sys.argv) > 1:
    act=str(sys.argv[1])

(X_train, y_train), (X_test, y_test) = mnist.load_data()

n_train = X_train.shape[0]
n_test = X_test.shape[0]

img_height = X_train.shape[1]
img_width  = X_train.shape[2]

X_train = X_train.reshape((n_train, img_width*img_height))
X_test  = X_test.reshape((n_test, img_width*img_height))

y_train = to_categorical(y_train, num_classes=10)
y_test  = to_categorical(y_test, num_classes=10)

mean = X_train.mean(axis=0)
std  = X_train.std(axis=0) + 1e-5

num_classes = 10
xi = Input(shape=(img_width*img_height,), name="input")
xl = Lambda(lambda image, mu, std: (image-mu)/std, arguments={'mu': mean, 'std': std})(xi)
xo = Dense(num_classes, name='y')(xl)
yo = Activation(act, name='y_act')(xo)
model = Model(inputs=[xi], outputs=[yo])

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=128,
          epochs=20,
          verbose=1,
          validation_split=0.1)

score = model.evaluate(X_test, y_test, verbose=0)

print('Loss: ', score[0])
print('Accuracy: ', score[1])
