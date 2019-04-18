import tensorflow as tf
import matplotlib.pyplot as plt
print(tf.__version__)

# get the data
fashion_mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = fashion_mnist.load_data()

# plt.imshow(training_images[0])
# plt.show(block=True)
# plt.interactive(False)

# print(training_labels[0])
# print(training_images[0])

# Normalization
training_images = training_images / 255
test_images = test_images / 255

# Model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

# Training with training data
model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=5)

# Evaluation with test data
model.evaluate(test_images, test_labels)
