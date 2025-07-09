# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 14:51:35 2021

@author: malth
"""
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from keras.preprocessing import image


model = tf.keras.models.Sequential([

    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    # The second convolution
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # The third convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),  # 512 neuron hidden layer
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for ('normal') clas and 1 for ('pneumonia') class
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# to get the summary of the model
model.summary()

# configure the model for traning by adding metrics
model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale = 1/255)
test_datagen = ImageDataGenerator(rescale = 1/255)

train_generator = train_datagen.flow_from_directory(
    'data/training',
    target_size = (300,300),
    batch_size = 16,
    class_mode = 'binary'
)

validation_generator = test_datagen.flow_from_directory(
    'data/validation',
    target_size = (300, 300),
    batch_size = 16,
    class_mode = 'binary'
)

# training the model
history = model.fit(
    train_generator,
    steps_per_epoch = 10,
    epochs = 10,
    validation_data = validation_generator
)


#Plot
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)

#’bo’ is for blue dot, ‘b’ is for solid blue line
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# load new unseen dataset
eval_datagen = ImageDataGenerator(rescale = 1/255)

test_generator = eval_datagen.flow_from_directory(
    'data/testing',
    target_size = (300, 300),
    batch_size = 16,
    class_mode = 'binary'
)

eval_result = model.evaluate_generator(test_generator, 624)
print('loss rate at evaluation data :', eval_result[0])
print('accuracy rate at evaluation data :', eval_result[1])
#%%
import os, shutil
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from keras.preprocessing import image
#from google.colab import files

input_layer = layers.Input(shape=(300, 300, 3))
con1 = layers.Conv2D(16, (2, 2), activation='relu')(input_layer)
pool1 = layers.MaxPool2D(2, 2)(con1)
con2 = layers.Conv2D(32, (2, 2), activation='relu')(pool1)
pool2 = layers.MaxPool2D(2, 2)(con2)
con3 = layers.Conv2D(64, (2, 2), activation='relu')(pool2)
pool3 = layers.MaxPool2D(2, 2)(con3)
con4 = layers.Conv2D(64, (2, 2), activation='relu')(pool3)
pool4 = layers.MaxPool2D(2, 2)(con4)
con5 = layers.Conv2D(64, (2, 2), activation='relu')(pool4)
pool5 = layers.MaxPool2D(2, 2)(con5)
flat1 = layers.Flatten()(pool5)
dense1 = layers.Dense(512, activation='relu')(flat1)
dense2 = layers.Dense(1, activation='sigmoid')(dense1)

model = Model(inputs=input_layer, outputs=dense2)


# to get the summary of the model
model.summary()
 
# configure the model for traning by adding metrics
model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['accuracy'])


train_datagen = ImageDataGenerator(rescale = 1/255)
test_datagen = ImageDataGenerator(rescale = 1/255)

train_generator = train_datagen.flow_from_directory(
    'data/training',
    target_size = (300,300),
    batch_size = 32,
    class_mode = 'binary'
)

validation_generator = test_datagen.flow_from_directory(
    'data/validation',
    target_size = (300, 300),
    batch_size = 64,
    class_mode = 'binary'
)

# training the model
history = model.fit(
    train_generator,
    steps_per_epoch = 10,
    epochs = 10,
    validation_data = validation_generator
)

test_datagen = ImageDataGenerator(rescale = 1/255)

test_generator = test_datagen.flow_from_directory(
    'data/testing',
    target_size = (300, 300),
    batch_size = 64,
    class_mode = 'binary'
)

eval_result = model.evaluate_generator(test_generator, 624)
print('loss rate at evaluation data :', eval_result[0])
print('accuracy rate at evaluation data :', eval_result[1])

model.summary()


#split til 2 mapper, mangler random
# root_path = (r'C:/Users/benja/PycharmProjects/final/data')
#
# folders = ['normal', 'pneumonia']
#
# for folder in folders:
#     os.mkdir(os.path.join(root_path, folder))
#
# dest1 = (r'C:/Users/benja/PycharmProjects/final/data\normal')
# dest2 = (r'C:/Users/benja/PycharmProjects/final/data\pneumonia')
#
# source = os.listdir(root_path)
#
# for files in source:
#     if (files.endswith('pneumonia.jpg')):
#         shutil.move(os.path.join(root_path, files), dest2)
#     if (files.endswith('normal.jpg')):
#         shutil.move(os.path.join(root_path, files), dest1)
#
# for _ in 'data/normal'/2:
#     i = random.choice('data/normal')