from os import path, getcwd

import tensorflow as tf

path = f"{getcwd()}/../mnist.npz"


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        if logs.get('acc') >= 0.998:
            print('\nReached desired accuracy (99.8%), No more training.')
            self.model.stop_training = True


def train_mnist_conv():
    mCallBack = myCallback()
    nmist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = nmist.load_data(path=path)
    x_train = x_train.reshape(60000,28,28,1)
    x_train = x_train / 255.0
    x_test = x_test.reshape(10000,28,28,1)
    x_test = x_test / 255.0
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(28,28,1)),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])
    history = model.fit(x_train, y_train, epochs=10, callbacks=[mCallBack])
    return history.epoch, history.history['acc'][-1]


train_mnist_conv()
