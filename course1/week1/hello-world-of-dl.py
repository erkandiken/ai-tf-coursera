import tensorflow as tf
import numpy as np
from tensorflow import keras

# model of one layer, one neuron and one input feature
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

# y = 2x - 1
xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
model.fit(xs, ys, epochs=500, verbose=0)

print('Weights: {}'.format(model.get_weights()))
print(model.predict([10, 14.0, -6.0]))
