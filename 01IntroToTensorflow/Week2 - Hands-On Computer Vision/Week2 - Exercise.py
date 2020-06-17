from os import path, getcwd

import tensorflow as tf

path = f"{getcwd()}/../mnist.npz"


class StopTrainingCaLLback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        if logs.get('acc') >= 0.99:
            print('\nReached desired accuracy (0.99), No more training.')
            self.model.stop_training = True


def train_mnist():
    mCallBack = StopTrainingCaLLback()
    nmist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = nmist.load_data(path=path)
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])
    history = model.fit(x_train, y_train, epochs=10, callbacks=[mCallBack])
    return history.epoch, history.history['acc'][-1]


train_mnist()
