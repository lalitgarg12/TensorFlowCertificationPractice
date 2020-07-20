import tensorflow_hub as hub
import tensorflow as tf
from tensorflow import keras

hub_layer = hub.KerasLayer("https://tfhub.dev/google/tf2-preview/nnlm-en-dim50/1",
                           output_shape=[50], input_shape=[], dtype=tf.string)

model = keras.Sequential()
model.add(hub_layer)
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.summary()

sentences = tf.constant(["It was a great movie", "The actors were amazing"])
embeddings = hub_layer(sentences)
print(embeddings)