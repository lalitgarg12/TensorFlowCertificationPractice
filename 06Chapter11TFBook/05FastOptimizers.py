
import tensorflow as tf
from tensorflow import  keras

#Momentum Optimizers
optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9)

#Nesterov Accelerated Gradient
optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)

#Adagrad
optimizer = keras.optimizers.Adagrad(lr=0.001)

#RMSProp
optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9)

#Adam Optimization
optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

#Adamax Optimization
optimizer = keras.optimizers.Adamax(lr=0.001, beta_1=0.9, beta_2=0.999)

#Nadam Optimization
optimizer = keras.optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999)
