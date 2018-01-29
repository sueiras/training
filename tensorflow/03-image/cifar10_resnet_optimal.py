from __future__ import print_function

#Basic libraries
import numpy as np
import tensorflow as tf
import time
from os import listdir
from os.path import isfile, join
import random

# Select GPU
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

data_path = '/home/ubuntu/data/training/image/cifar10/'


from tensorflow.contrib.keras import models, layers, optimizers, callbacks, preprocessing, regularizers


# Based on https://gist.github.com/JefferyRPrice/c1ecc3d67068c8d9b3120475baba1d7e

def residual_layer(input_tensor, nb_in_filters=64, nb_bottleneck_filters=16, filter_sz=3, stage=0, reg=0.0):

    bn_name = 'bn' + str(stage)
    conv_name = 'conv' + str(stage)
    relu_name = 'relu' + str(stage)
    merge_name = 'add' + str(stage)

    # batchnorm-relu-conv, from nb_in_filters to nb_bottleneck_filters via 1x1 conv
    if stage>1: # first activation is just after conv1
        x = layers.BatchNormalization(axis=-1, name=bn_name+'a')(input_tensor)
        x = layers.Activation('relu', name=relu_name+'a')(x)
    else:
        x = input_tensor

    x = layers.Conv2D(nb_bottleneck_filters, (1, 1),
                      kernel_initializer='glorot_normal',
                      kernel_regularizer=regularizers.l2(reg),
                      use_bias=False,
                      name=conv_name+'a')(x)

    # batchnorm-relu-conv, from nb_bottleneck_filters to nb_bottleneck_filters via FxF conv
    x = layers.BatchNormalization(axis=-1, name=bn_name+'b')(x)
    x = layers.Activation('relu', name=relu_name+'b')(x)
    x = layers.Conv2D(nb_bottleneck_filters, (filter_sz, filter_sz),
                      padding='same',
                      kernel_initializer='glorot_normal',
                      kernel_regularizer=regularizers.l2(reg),
                      use_bias = False,
                      name=conv_name+'b')(x)

    # batchnorm-relu-conv, from nb_in_filters to nb_bottleneck_filters via 1x1 conv
    x = layers.BatchNormalization(axis=-1, name=bn_name+'c')(x)
    x = layers.Activation('relu', name=relu_name+'c')(x)
    x = layers.Conv2D(nb_in_filters, (1, 1),
                      kernel_initializer='glorot_normal',
                      kernel_regularizer=regularizers.l2(reg),
                      name=conv_name+'c')(x)

    # merge
    x = layers.add([x, input_tensor], name=merge_name)

    return x


# Basic
#sz_ly0_filters, nb_ly0_filters, nb_ly0_stride = (64,5,2)
#sz_res_filters, nb_res_filters, nb_res_stages = (3,16,3)


# 92% of accuracy
sz_ly0_filters, nb_ly0_filters, nb_ly0_stride = (128,3,2)
sz_res_filters, nb_res_filters, nb_res_stages = (3,32,25)


  
    
img_input = layers.Input(shape=(32,32,3), name='cifar')

# Initial layers
x = layers.Conv2D(sz_ly0_filters, (nb_ly0_filters,nb_ly0_filters),
                  strides=(nb_ly0_stride, nb_ly0_stride), padding='same', 
                  kernel_initializer='glorot_normal',
                  kernel_regularizer=regularizers.l2(1.e-4),
                  use_bias=False, name='conv0')(img_input)

x = layers.BatchNormalization(axis=-1, name='bn0')(x)
x = layers.Activation('relu', name='relu0')(x)

# Resnet layers
for stage in range(1, nb_res_stages+1):
    x = residual_layer(x, 
                       nb_in_filters=sz_ly0_filters,
                       nb_bottleneck_filters=nb_res_filters,
                       filter_sz=sz_res_filters, 
                       stage=stage,
                       reg=0.0)

# Complete last resnet layer    
x = layers.BatchNormalization(axis=-1, name='bnF')(x)
x = layers.Activation('relu', name='reluF')(x)


# Final layer
x = layers.AveragePooling2D((16, 16), name='avg_pool')(x)
x = layers.Flatten(name='flat')(x)
x = layers.Dense(10, activation='softmax', name='fc1')(x)

model1 = models.Model(inputs=img_input, outputs=x)
model1.summary()


my_datagen = preprocessing.image.ImageDataGenerator()


# Augmentation for training
train_datagen = preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True)

# Augmentation configuration we will use for testing:
# only rescaling
test_datagen = preprocessing.image.ImageDataGenerator(rescale=1. / 255)


train_generator = train_datagen.flow_from_directory(
    join(data_path, 'train'),
    target_size=(32, 32),
    batch_size=32)


test_generator = test_datagen.flow_from_directory(
    join(data_path, 'test'),
    target_size=(32, 32),
    batch_size=32)



tb_callback_ln = callbacks.TensorBoard(log_dir='/tmp/tensorboard/cifar10/resnet2')

batch_size = 32
nb_train_samples = 50000
nb_test_samples = 10000

opt = optimizers.Adam(lr=1E-3)
model1.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
history1 = model1.fit_generator(
    train_generator,
    steps_per_epoch = nb_train_samples // batch_size,
    epochs = 50,
    validation_data = test_generator,
    validation_steps = nb_test_samples // batch_size,
    callbacks=[tb_callback_ln]
    )
    
opt = optimizers.Adam(lr=1E-4)
model1.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
history2 = model1.fit_generator(
    train_generator,
    steps_per_epoch = nb_train_samples // batch_size,
    epochs = 25,
    validation_data = test_generator,
    validation_steps = nb_test_samples // batch_size,
    callbacks=[tb_callback_ln]
    )

opt = optimizers.Adam(lr=1E-5)
model1.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
history3 = model1.fit_generator(
    train_generator,
    steps_per_epoch = nb_train_samples // batch_size,
    epochs = 25,
    validation_data = test_generator,
    validation_steps = nb_test_samples // batch_size,
    callbacks=[tb_callback_ln]
    )

# Save model
model1.save('cifar10_resnet_optimal.h5')



