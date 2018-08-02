
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image

import keras
import tensorflow
import cv2
import numpy as np
from face_network import create_face_network

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image

from keras import preprocessing

from keras.preprocessing import sequence
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K


'''

img = load_img(image) # this is a PIL image
print(img)

array = img_to_array(img)
    
arrayresized = cv2.resize(array, (150,150))
print(arrayresized) # this is a Numpy array with shape (3, 150, 150)


'''


img_width, img_height = 150, 150

train_data_dir = 'C:/data/train'
validation_data_dir = 'C:/data/validation'
nb_train_samples = 1500
nb_validation_samples = 650
epochs = 20
batch_size = 15

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:

# only rescaling

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

model.save('train.h5')
model.save_weights('train.h5')

image_path = './joe-biden.jpg'

img = load_img(image_path) # this is a PIL image
print(img)

array = img_to_array(img)
    
arrayresized = cv2.resize(array, (150,150))
print(arrayresized) # this is a Numpy array with shape (3, 150, 150)

inputarray = arrayresized[np.newaxis,...] # dimension added to fit input size

prediction = model.predict_proba(inputarray)

print(prediction)