# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 12:36:37 2019

@author: Talha
"""
#import libraries
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import Activation, Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.optimizers import Adam,RMSprop

path='D:/Datasets/Image dataset/eye disease/eye-disease-dataset/'

image_shape = (100,100,3) #size of images to feed in neural network
#data augmentation

datagen = ImageDataGenerator(rotation_range=30, # rotate the image 30 degrees
                               width_shift_range=0.1, # Shift the pic width by a max of 10%
                               height_shift_range=0.1, # Shift the pic height by a max of 10%
                               rescale=1/255, # Rescale the image by normalzing it.
                               shear_range=0.2, # Shear means cutting away part of the image (max 20%)
                               zoom_range=0.2, # Zoom in by 20% max
                               horizontal_flip=True, # Allo horizontal flipping
                               vertical_flip=True,
                               fill_mode='nearest',# Fill in missing pixels with the nearest filled value
                               validation_split=0.2#split data to train and test
                              )

#here we dont have seprate tain and test folder, so I split my folder to train and test in ImageDataGenerator (line 32 )


#load the training data
train_generator = datagen.flow_from_directory(
    path+'Cropped_Images/',
    target_size=image_shape[0:2],
    batch_size=100,
    class_mode='categorical',
    subset='training')

#load the test data
test_generator = datagen.flow_from_directory(
    path+'Cropped_Images/',
    target_size=image_shape[0:2],
    batch_size=100,
    class_mode='categorical',
    subset='validation')


#create model
model = Sequential()
model.add(Conv2D(filters = 16, kernel_size = 3,input_shape=image_shape,padding='same'))
model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters = 32,kernel_size = 3,activation= 'relu',padding='same'))
#model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters = 64,kernel_size = 3,activation= 'relu',padding='same'))
#model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(5,activation = 'softmax'))

model.summary()

#compile the model
opt = RMSprop(lr=0.0001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

#create class weights for unbalance dataset
from sklearn.utils import class_weight
y_train=train_generator.classes
class_weights = class_weight.compute_class_weight('balanced',np.unique(y_train),y_train)
class_weights=dict(enumerate(class_weights))


#train model
results = model.fit_generator(train_generator,epochs=10,
                              steps_per_epoch=30,
                              validation_data=test_generator,
                             validation_steps=30,
                             class_weight=class_weights)

#print classification report
y_true=test_generator.classes
class_label=list(train_generator.class_indices.keys())
from sklearn.metrics import classification_report
y_pred = model.predict_generator(test_generator,steps=186/100) #186 are total example in test set and 40 is batch size
y_pred = np.argmax(y_pred, axis=1)
print(classification_report(y_true, y_pred,target_names=class_label))



K.clear_session()




