import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers

import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
nombre_modelo = 'prueba_gru_multiples_gpus'

(x_train, y_train), (x_test, y_test) = mnist.load_data()

num_labels = len(np.unique(y_train))

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

image_size = x_train.shape[1]
x_train = np.reshape(x_train, [-1, image_size, image_size])
x_test = np.reshape(x_test, [-1, image_size, image_size])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

input_shape = (image_size, image_size)
units = 256
dropout = 0.2
learning_rate = 0.001


GPUs = [2]
GPUs = ["/gpu:{}".format(gpu) for gpu in GPUs]
batch_size = 256*len(GPUs)


strategy = tf.distribute.MirroredStrategy(GPUs)
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

with strategy.scope():
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(GRU(units=units, dropout=dropout, use_bias=False, return_sequences=True))
    model.add(GRU(units=units, dropout=dropout, use_bias=False, go_backwards=True))
    model.add(Dense(num_labels))
    model.add(Activation('softmax'))
    model.summary()

    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=2, validation_data=(x_test, y_test))
loss, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print("\nTest accuracy: %.1f%%" % (100.0 * acc))
