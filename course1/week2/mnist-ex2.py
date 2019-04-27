import tensorflow as tf
print(tf.__version__)


class MyCallbacks(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        acc_threshold = 0.99
        accuracy = logs.get('accuracy')
        if accuracy > acc_threshold:
            print('\nReached {0} % accuracy so cancelling training!\n'.format(acc_threshold))
            self.model.stop_training = True

        loss_threshold = 0.001
        loss = logs.get('loss')
        if loss < loss_threshold:
            print('\nReached {0} loss so cancelling training!\n'.format(loss_threshold))
            self.model.stop_training = True


# load train and test data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalization
x_train, x_test = x_train / 255.0,  x_test / 255.0

# model
model = tf.keras.Sequential([tf.keras.layers.Flatten(),
                             tf.keras.layers.Dense(512, activation=tf.nn.relu),
                             tf.keras.layers.Dense(10, activation=tf.nn.softmax)
                             ])
# compile the model and fit data
model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, callbacks=[MyCallbacks()])

# evaluate on test data
model.evaluate(x_test, y_test)





