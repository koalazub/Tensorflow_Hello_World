from __future__ import absolute_import, division, print_function, unicode_literals

# import tf and tf.keras
from array import array

import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

print(tf.version)

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
               'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

train_images.shape

len(train_labels)

train_labels

test_images.shape

len(test_labels)

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# Normalise to 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# displaying first 25 images

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    # show the images and convert to black and white
    plt.imshow(train_images[i])
    # classify each item to a label
    plt.xlabel(class_names[train_labels[i]])
    plt.show()

# configuration of neural network and compilation of model

model = keras.Sequential([
    # transform from
    keras.layers.Flatten(input_shape=(28, 28)),
    # densely connect to two layers
    # this has 128 layers
    keras.layers.Dense(128, activation=tf.nn.relu),
    # feed in K(data) normalise into probability distribution
    # softmax later returns probability score pertaining to one of 10 classes set by me
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
# update based on data seen via loss function
model.compile(optimizer='adam',
              # measure accuracy based on training. minimise 'steering' the model in the right
              # direction
              loss='sparse_categorical_crossentropy',
              # monitor training and testing steps
              metrics=['accuracy'])
# training function
model.fit(train_images, train_labels, epochs=5)

# Evaluate accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('test  accuracy: ', test_acc)

# Make predictions
predictions = model.predict(test_images)

# predicted labels
indexPred = predictions[0]

np.argmax(indexPred)

# TODO if this doesn't work remove argmax
np.argmax(test_labels[0])


# graphing full set of predictions
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

        plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                             100 * np.max(predictions_array),
                                             class_names[true_label]),
                   color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color='#777777')
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[true_label].set_color('blue')
    thisplot[predicted_label].set_color('red')


# Peeking at 0th image

i = 0
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions, test_labels)
plt.show()

i = 12
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions, test_labels)
plt.show()
# plot first X images, predicted label, and true label
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions, test_labels)
    plt.show()

    # grab image from data set

    img = test_images[0]
    print(img.shape)

    img = (np.expand_dims(img, 0))
    print(img.shape)

    # predict image

    predictions_single = model.predict(img)
    print(predictions_single)

    plot_value_array(0, predictions_single, test_labels)
    plt.xticks(range(10), class_names, rotation=45)
    plt.show()

    prediction_result = np.argmax(predictions_single[0])
    print(prediction_result)
