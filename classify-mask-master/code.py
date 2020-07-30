# -*- coding: utf-8 -*-
"""
Created on Tue May  5 14:58:44 2020

@author: DELL
"""

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()
#padding='same  keep same dim of i/p,paddding='valid' element extra dim  so loss of data happens
#link  https://www.pico.net/kb/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-tensorflow
#fine tuning  1st half freeze remaining trainable
# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu')) #3 is no of channels ,64 is dim of 2d array in each channel
#32 is fliters(feature detector), 3,3 is row and col of feature detector
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2))) #without these get large features then very high compattue but  not loose performance and info

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu')) #already know size, already have so no need I/P
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2, #add new images ,best performance reduce overfitting
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('data',
                                                 target_size = (64, 64), #T_S HIGH MORE ACC BECAUSE get more info(lot of pixels)
                                                 batch_size = 32,
                                                 class_mode = 'binary')

val_set = test_datagen.flow_from_directory('dest_folder/val',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 4000,
                         epochs = 2,
                         validation_data = val_set,
                         validation_steps = 2000)

