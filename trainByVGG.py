# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 22:19:22 2018

@author: Administrator
"""

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint, Callback, TensorBoard
from keras.applications.vgg16 import VGG16
from keras.models import Model
import numpy as np
import cv2
import os

ishape = 224

train_data_dir = './train'
validation_data_dir = './validation'
nb_train_samples = 1000
nb_validation_samples = 60
epochs = 5
batch_size = 1

model_vgg=VGG16(include_top=False,weights='imagenet',input_shape=(ishape,ishape,3))
for layer in model_vgg.layers:
    layer.trainable=False
model=Flatten()(model_vgg.output)
model=Dense(10,activation='softmax')(model)
model_vgg_mnist_pretrain=Model(model_vgg.input,model,name='vgg16_pretrain')

model_vgg_mnist_pretrain.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
validation_datagen = ImageDataGenerator(
    rescale=1. / 255)

train_generator=train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(ishape, ishape),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator=validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(ishape, ishape),
    batch_size=batch_size,
    class_mode='categorical')


filepath="1.weights-improvement-{epoch:02d}-{val_acc:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
tensorboard = TensorBoard(log_dir='logs', histogram_freq=0)
callbacks_list = [checkpoint, tensorboard]


model_vgg_mnist_pretrain.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    callbacks=callbacks_list)