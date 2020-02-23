# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 09:55:25 2020

@author: hinaa
"""

# Convolutional Neural Network


# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense,Dropout
# Initialising the CNN
classifier = Sequential()
 
#1st convolution layer
classifier.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48,48,3)))
classifier.add(MaxPooling2D(pool_size=(5,5), strides=(2, 2)))
 
#2nd convolution layer
classifier.add(Conv2D(64, (3, 3), activation='relu'))

classifier.add(MaxPooling2D(pool_size=(3,3), strides=(2, 2)))
 
#3rd convolution layer
classifier.add(Conv2D(128, (3, 3), activation='relu'))

classifier.add(MaxPooling2D(pool_size=(3,3), strides=(2, 2)))
 
classifier.add(Flatten())
 
#fully connected neural networks
# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units =3, activation = 'softmax'))


 
# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('images/train',#folder names
                                                 target_size = (48,48),
                                                 batch_size = 100,  #16095/100
                                                 class_mode = 'categorical')#train set    

test_set = test_datagen.flow_from_directory('images/validation',
                                            target_size = (48,48),
                                            batch_size = 100, 
                                            class_mode = 'categorical')#test set

classifier.fit_generator(training_set,
                         samples_per_epoch = 16095,#total  train set i have 351 happy,sad and angry train added
                         nb_epoch =10,
                         validation_data = test_set,
                         nb_val_samples = 3924)#how many samples in test happy sad and anry added
classifier.save('expressions1.h5')


   