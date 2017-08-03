import os
import sys
sys.path.insert(1, '/home/rbbidart/.local-mosaic/lib/python2.7/site-packages')

import numpy as np
import pandas as pd
import glob
import random
from random import randint
from PIL import Image

import keras
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
from keras import layers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Reshape, Input
from keras.layers.core import Activation, Dense, Lambda
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization


def data_generator_no_aug(file_loc, batch_size):
    bad_files = []
    while 1:
        all_files=glob.glob(os.path.join(file_loc, '*'))
        random.shuffle(all_files) # randomize after every epoch
        num_batches = int(np.floor(len(all_files)/batch_size))

        for batch in range(num_batches):
            x=[]
            y=[]
            batch_files = all_files[batch_size*batch:batch_size*(batch+1)]
            for image_loc in batch_files:
                if (image_loc.rsplit('_', 1)[-1] in bad_files):
                    pass
                else:
                    image = Image.open(image_loc)
                    width, height = image.size
                    image = np.reshape(np.array(image.getdata()), (height, width, 3))
                    image = image/255.0

                    y_temp = np.array(int(image_loc.rsplit('/', 1)[-1].split('_', 1)[0]))
                    y_temp = np.eye(4)[y_temp]

                    x.append(image)
                    y.append(y_temp)

            x=np.array(x)
            y=np.array(y)
            yield (x, y)


def data_gen_aug(file_loc, batch_size, image_shape=(32, 32), square_rot_p=.3, translate = 3):
    # square_rot_p is the prob of using a 90x rotation, otherwise sample from 360. Possibly not useful
    # translate is maximum number of pixels to translate by. Make it close the doctor's variance in annotation
    square_rot_p = int(square_rot_p)
    translate = int(translate) 

    while 1:
        all_files=glob.glob(os.path.join(file_loc, '*'))
        random.shuffle(all_files) # randomize after every epoch
        num_batches = int(np.floor(len(all_files)/batch_size))

        for batch in range(num_batches):
            x=[]
            y=[]
            batch_files = all_files[batch_size*batch:batch_size*(batch+1)]
            for image_loc in batch_files:
                image = Image.open(image_loc)

                # All the randomness:
                ts_sz_row = randint(-1*translate, translate)
                ts_sz_col = randint(-1*translate, translate)
                angle = np.random.uniform(0, 360,1)
                flip_vert = random.randint(0, 1)
                flip_hor =random.randint(0, 1)

                # APPLY AUGMENTATION:
                # flips
                if flip_vert:
                    image = image.transpose(Image.FLIP_TOP_BOTTOM)
                if flip_hor:
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)

                # rotation
                square_rot =  bool((np.random.uniform(0, 1, 1)<square_rot_p))
                if square_rot:  # maybe this is dumb, but it cant hurt
                    rotations=['0', '90', '180', '270']
                    angle = int(random.choice(rotations))
                    image=image.rotate(angle)
                else:
                    image=image.rotate(angle)
                image = image.resize(image_shape)

                # translate
                width, height = image.size
                image = np.reshape(np.array(image.getdata()), (height, width, 3))
                image_new = np.zeros((height, width, 3))
                # If translate is negative it moves left
                image_new[max(ts_sz_row, 0):height+max(ts_sz_row, 0), max(ts_sz_col, 0):width+max(ts_sz_col, 0), :] = image[max(ts_sz_row, 0):height+max(ts_sz_row, 0), max(ts_sz_col, 0):width+max(ts_sz_col, 0), :]
                image = image_new

                image = image/255.0 # make pixels in [0,1]        
                y_temp = np.array(int(image_loc.rsplit('/', 1)[-1].split('_', 1)[0]))
                y_temp = np.eye(4)[y_temp]
                
                x.append(image)
                y.append(y_temp)
            
            x=np.array(x)
            y=np.array(y)
            yield (x, y)

###############

# def create_heatmap(image_loc, model, stride = 4):
#     image = np.array(Image.open(image_loc))
#     image = image/255.0 # During training the images were normalized
#     height = int(32)
#     stride = int(stride)
    
#     out_size = (np.floor(np.array(image.shape)/stride)).astype(int)
#     out_size[2] = 4 # there are 4 classes
#     out_image = np.zeros(out_size.astype(int))
#     print('out_image.shape', out_image.shape)

#     delta=int((height)/2)
#     image = np.lib.pad(image, ((delta, delta), (delta, delta), (0,0)), 'constant', constant_values=(0, 0))

#     # Double for loop is hideous and slow. Probably should be done in GPU somehow.
#     for row in range(0, out_image.shape[0], 1):
#         for col in range(0, out_image.shape[1], 1):
#             in_col = int(col*stride+delta)
#             in_row = int(row*stride+delta)
#             seg_image = image[in_row-delta:in_row+delta, in_col-delta:in_col+delta,:]
#             seg_image = np.expand_dims(seg_image, axis=0) # keras expects batchsize as index 0
#             pred = model.predict(seg_image, batch_size=1, verbose=0)
#             out_image[row, col, :] = pred
#     return out_image

###############


def conv1(learning_rate = .001, dropout = .5, im_size=32):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(im_size, im_size, 3), kernel_initializer='he_normal'))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(im_size, im_size, 32), kernel_initializer='he_normal'))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))

    model.add(Conv2D(16, (3, 3), padding='same', input_shape=(im_size, im_size, 32), kernel_initializer='he_normal'))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    
    model.add(Flatten())
    model.add(Dense(512, kernel_initializer="he_normal"))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))

    model.add(Dense(4, kernel_initializer="he_normal"))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Activation('softmax'))

    Adam = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss="categorical_crossentropy", optimizer=Adam, metrics=['accuracy'])
    return model

def conv2(learning_rate = .002, dropout = .5, im_size=32):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(im_size, im_size, 3), kernel_initializer='he_normal'))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(im_size, im_size, 32), kernel_initializer='he_normal'))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    
    model.add(Conv2D(16, (3, 3), padding='same', input_shape=(im_size, im_size, 32), kernel_initializer='he_normal'))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))

    model.add(Conv2D(8, (3, 3), padding='same', input_shape=(im_size, im_size, 16), kernel_initializer='he_normal'))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    
    model.add(Flatten())
    model.add(Dense(1024, kernel_initializer="he_normal"))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))

    model.add(Dense(4, kernel_initializer="he_normal"))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Activation('softmax'))

    Adam = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss="categorical_crossentropy", optimizer=Adam, metrics=['accuracy'])
    return model

#### Fully convolutional

# Custom functions to use because keras flatten doesn't like variable sized input
def flatten(x):
    batch_size = K.shape(x)[0]
    x = K.reshape(x, (batch_size, -1)) 
    return x

def conv_fc1(learning_rate = .001, dropout = .2, im_size=32):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(None, None, 3), kernel_initializer='he_normal'))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))

    model.add(Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal'))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))

    model.add(Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal'))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Activation('relu')) 
    model.add(Dropout(dropout))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # to make it fully convolutional we use a convolution layer to reshape rather than fully connected
    # Valid padding instead of using global average pooling. We want the class for the center, so shouldnt average overall class probs
    model.add(Conv2D(4, (int(im_size/2), int(im_size/2)), padding='valid', kernel_initializer='he_normal'))
    #model.add(Reshape((None, 1, 1, 4)))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Dropout(dropout))

    model.add(Lambda(flatten))
    model.add(Activation('softmax'))

    Adam = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss="categorical_crossentropy", optimizer=Adam, metrics=['accuracy'])
    return model

def conv_fc2(learning_rate = .001, dropout = .2, im_size=32):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(None, None, 3), kernel_initializer='he_normal'))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))

    model.add(Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal'))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))

    model.add(Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal'))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    
    # to make it fully convolutional we use a convolution layer to reshape rather than fully connected
    # Valid padding instead of using global average pooling. We want the class for the center, so shouldnt average overall class probs
    model.add(Conv2D(4, (int(im_size), int(im_size)), padding='valid', kernel_initializer='he_normal'))
    #model.add(Reshape((None, 1, 1, 4)))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Dropout(dropout))

    model.add(Lambda(flatten))
    model.add(Activation('softmax'))

    Adam = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss="categorical_crossentropy", optimizer=Adam, metrics=['accuracy'])
    return model


def conv_fc3(learning_rate = .001, dropout = .2, im_size=32):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(None, None, 3), kernel_initializer='he_normal'))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))

    model.add(Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal'))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))

    model.add(Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal'))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    
    # to make it fully convolutional we use a convolution layer to reshape rather than fully connected
    # Valid padding instead of using global average pooling. We want the class for the center, so shouldnt average overall class probs
    model.add(Conv2D(256, (int(im_size), int(im_size)), padding='valid', kernel_initializer='he_normal'))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Dropout(dropout))

    model.add(Conv2D(4, (1, 1), padding='valid', kernel_initializer='he_normal'))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Dropout(dropout))

    model.add(Lambda(flatten))
    model.add(Activation('softmax'))

    Adam = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss="categorical_crossentropy", optimizer=Adam, metrics=['accuracy'])
    return model


def conv_fc4(learning_rate = .001, dropout = .2, im_size=32):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(None, None, 3), kernel_initializer='he_normal'))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))

    model.add(Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal'))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))

    model.add(Conv2D(16, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal'))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    
    # to make it fully convolutional we use a convolution layer to reshape rather than fully connected
    # Valid padding instead of using global average pooling. We want the class for the center, so shouldnt average overall class probs
    model.add(Conv2D(4, (int(im_size/2), int(im_size/2)), padding='valid', kernel_initializer='he_normal'))
    #model.add(Reshape((None, 1, 1, 4)))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Dropout(dropout))

    model.add(Lambda(flatten))
    model.add(Activation('softmax'))

    Adam = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss="categorical_crossentropy", optimizer=Adam, metrics=['accuracy'])
    return model


def conv_fc5(learning_rate = .001, dropout = .2, im_size=32):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(None, None, 3), kernel_initializer='he_normal'))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))

    model.add(Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal'))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))

    model.add(Conv2D(16, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal'))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    
    # to make it fully convolutional we use a convolution layer to reshape rather than fully connected
    # Valid padding instead of using global average pooling. We want the class for the center, so shouldnt average overall class probs
    model.add(Conv2D(16, (int(im_size/2), int(im_size/2)), padding='valid', kernel_initializer='he_normal'))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Dropout(dropout))

    model.add(Conv2D(4, (1, 1), padding='valid', kernel_initializer='he_normal'))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Dropout(dropout))

    model.add(Lambda(flatten))
    model.add(Activation('softmax'))

    Adam = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss="categorical_crossentropy", optimizer=Adam, metrics=['accuracy'])
    return model


def conv_fc6(learning_rate = .001, dropout = .2, im_size=32):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(None, None, 3), kernel_initializer='he_normal'))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))

    model.add(Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal'))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))

    model.add(Conv2D(16, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal'))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))

    model.add(Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal'))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    
    # to make it fully convolutional we use a convolution layer to reshape rather than fully connected
    # Valid padding instead of using global average pooling. We want the class for the center, so shouldnt average overall class probs
    model.add(Conv2D(4, (int(im_size/2), int(im_size/2)), padding='valid', kernel_initializer='he_normal'))
    #model.add(Reshape((None, 1, 1, 4)))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Dropout(dropout))

    model.add(Lambda(flatten))
    model.add(Activation('softmax'))

    Adam = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss="categorical_crossentropy", optimizer=Adam, metrics=['accuracy'])
    return model

def conv_fc7(learning_rate = .001, dropout = .2, im_size=32):
    input_shape = (None, None, 3)
    img_input = Input(shape=input_shape)

    x = conv2d_bn(img_input, 32, 3, 3, dropout = dropout, padding='same', strides=(1, 1))
    x = conv2d_bn(x, 32, 3, 3, dropout = dropout, padding='same', strides=(1, 1))
    x = conv2d_bn(x, 32, 3, 3, dropout = dropout, padding='same', strides=(2, 2))

    x = conv2d_bn(x, 32, 3, 3, dropout = dropout, padding='same', strides=(1, 1))
    x = conv2d_bn(x, 32, 3, 3, dropout = dropout, padding='same', strides=(1, 1))
    
    # to make it fully convolutional we use a convolution layer to reshape rather than fully connected
    # Valid padding instead of using global average pooling. We want the class for the center, so shouldnt average overall class probs
    x = conv2d_bn(x, 4, int(im_size/2), int(im_size/2), dropout = dropout, padding='valid', strides=(1, 1))

    x = Lambda(flatten)(x)
    x = Activation('softmax')(x)

    model = Model(img_input, x)
    Adam = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss="categorical_crossentropy", optimizer=Adam, metrics=['accuracy'])
    return model

def conv_fc8(learning_rate = .001, dropout = .2, im_size=32):
    input_shape = (None, None, 3)
    img_input = Input(shape=input_shape)

    x = conv2d_bn(img_input, 32, 3, 3, dropout = dropout, padding='same', strides=(1, 1))
    x = conv2d_bn(x, 32, 3, 3, dropout = dropout, padding='same', strides=(1, 1))
    x = conv2d_bn(x, 32, 3, 3, dropout = dropout, padding='same', strides=(2, 2))

    x = conv2d_bn(x, 32, 3, 3, dropout = dropout, padding='same', strides=(1, 1))
    x = conv2d_bn(x, 32, 3, 3, dropout = dropout, padding='same', strides=(1, 1))
    x = conv2d_bn(x, 32, 3, 3, dropout = dropout, padding='same', strides=(2, 2))

    # to make it fully convolutional we use a convolution layer to reshape rather than fully connected
    # Valid padding instead of using global average pooling. We want the class for the center, so shouldnt average overall class probs
    x = conv2d_bn(x, 4, int(im_size/4), int(im_size/4), dropout = dropout, padding='valid', strides=(1, 1))

    x = Lambda(flatten)(x)
    x = Activation('softmax')(x)

    model = Model(img_input, x)
    Adam = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss="categorical_crossentropy", optimizer=Adam, metrics=['accuracy'])
    return model

# Inception style models 
# to keep this concise. same as ^ but with dropout

def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              dropout, 
              padding='same',
              strides=(1, 1)):
    x = Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout)(x)
    return x

# This keeps the same filter numbers as the initial model, but is cut off after 2 inception blocks, with no downsammpling
# Try to be small size, but close to the original model
def conv_incp1(learning_rate = .001, dropout = .1, im_size=32):
    input_shape = (None, None, 3)
    img_input = Input(shape=input_shape)

    # Remove stride in the first layer, no max pool, 3 instead of 5
    x = conv2d_bn(img_input, 32, 3, 3, dropout = dropout, padding='same', strides=(1, 1))
    x = conv2d_bn(x, 32, 3, 3, dropout = dropout, padding='same', strides=(1, 1))
    x = conv2d_bn(x, 32, 3, 3, dropout = dropout, padding='same', strides=(2, 2))

    # Inception Blocks - Mixed 1
    branch1x1 = conv2d_bn(x, 64, 1, 1, dropout = dropout)

    branch5x5 = conv2d_bn(x, 48, 1, 1, dropout = dropout)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5, dropout = dropout)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1, dropout = dropout)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, dropout = dropout)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, dropout = dropout)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1, dropout = dropout)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool])

    # mixed 2
    branch1x1 = conv2d_bn(x, 64, 1, 1, dropout = dropout)

    branch5x5 = conv2d_bn(x, 48, 1, 1, dropout = dropout)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5, dropout = dropout)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1, dropout = dropout)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, dropout = dropout)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, dropout = dropout)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1, dropout = dropout)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool])

    # mixed 4
    branch1x1 = conv2d_bn(x, 192, 1, 1, dropout = dropout)

    branch7x7 = conv2d_bn(x, 128, 1, 1, dropout = dropout)
    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7, dropout = dropout)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1, dropout = dropout)

    branch7x7dbl = conv2d_bn(x, 128, 1, 1, dropout = dropout)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1, dropout = dropout)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7, dropout = dropout)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1, dropout = dropout)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7, dropout = dropout)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1, dropout = dropout)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool])

    x = conv2d_bn(img_input, 16, 3, 3, dropout = dropout, padding='same', strides=(1, 1))
    x = conv2d_bn(x, 4, int(im_size), int(im_size), dropout = dropout, padding='valid', strides=(1, 1))

    x = Lambda(flatten)(x)
    x = Activation('softmax')(x)

    model = Model(img_input, x)
    Adam = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss="categorical_crossentropy", optimizer=Adam, metrics=['accuracy'])
    return model

# Similair to conv incep1, but without downsampling of the image
def conv_incp2(learning_rate = .001, dropout = .1, im_size=32):
    input_shape = (None, None, 3)
    img_input = Input(shape=input_shape)

    # Remove stride in the first layer, no max pool, 3 instead of 5
    x = conv2d_bn(img_input, 32, 3, 3, dropout = dropout, padding='same', strides=(1, 1))
    x = conv2d_bn(x, 32, 3, 3, dropout = dropout, padding='same', strides=(1, 1))
    x = conv2d_bn(x, 32, 3, 3, dropout = dropout, padding='same', strides=(1, 1))

    # Inception Blocks - Mixed 1
    branch1x1 = conv2d_bn(x, 64, 1, 1, dropout = dropout)

    branch5x5 = conv2d_bn(x, 48, 1, 1, dropout = dropout)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5, dropout = dropout)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1, dropout = dropout)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, dropout = dropout)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, dropout = dropout)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1, dropout = dropout)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool])

    # mixed 2
    branch1x1 = conv2d_bn(x, 64, 1, 1, dropout = dropout)

    branch5x5 = conv2d_bn(x, 48, 1, 1, dropout = dropout)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5, dropout = dropout)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1, dropout = dropout)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, dropout = dropout)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, dropout = dropout)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1, dropout = dropout)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool])

    # mixed 4
    branch1x1 = conv2d_bn(x, 192, 1, 1, dropout = dropout)

    branch7x7 = conv2d_bn(x, 128, 1, 1, dropout = dropout)
    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7, dropout = dropout)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1, dropout = dropout)

    branch7x7dbl = conv2d_bn(x, 128, 1, 1, dropout = dropout)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1, dropout = dropout)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7, dropout = dropout)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1, dropout = dropout)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7, dropout = dropout)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1, dropout = dropout)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool])

    x = conv2d_bn(img_input, 16, 3, 3, dropout = dropout, padding='same', strides=(2, 2))
    x = conv2d_bn(x, 4, int(im_size/2), int(im_size/2), dropout = dropout, padding='valid', strides=(1, 1))

    x = Lambda(flatten)(x)
    x = Activation('softmax')(x)

    model = Model(img_input, x)
    Adam = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss="categorical_crossentropy", optimizer=Adam, metrics=['accuracy'])
    return model


# Smaller  one with downsampling at start
def conv_incp3(learning_rate = .001, dropout = .1, im_size=32):
    input_shape = (None, None, 3)
    img_input = Input(shape=input_shape)

    # Remove stride in the first layer, no max pool, 3 instead of 5
    x = conv2d_bn(img_input, 32, 3, 3, dropout = dropout, padding='same', strides=(1, 1))
    x = conv2d_bn(x, 32, 3, 3, dropout = dropout, padding='same', strides=(2, 2))

    # Inception Blocks - Mixed 1
    branch1x1 = conv2d_bn(x, 32, 1, 1, dropout = dropout)

    branch5x5 = conv2d_bn(x, 24, 1, 1, dropout = dropout)
    branch5x5 = conv2d_bn(branch5x5, 32, 5, 5, dropout = dropout)

    branch3x3dbl = conv2d_bn(x, 32, 1, 1, dropout = dropout)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 48, 3, 3, dropout = dropout)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 48, 3, 3, dropout = dropout)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 16, 1, 1, dropout = dropout)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool])

    # mixed 2
    branch1x1 = conv2d_bn(x, 16, 1, 1, dropout = dropout)

    branch5x5 = conv2d_bn(x, 12, 1, 1, dropout = dropout)
    branch5x5 = conv2d_bn(branch5x5, 32, 5, 5, dropout = dropout)

    branch3x3dbl = conv2d_bn(x, 16, 1, 1, dropout = dropout)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 24, 3, 3, dropout = dropout)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 24, 3, 3, dropout = dropout)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 16, 1, 1, dropout = dropout)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool])

    x = conv2d_bn(x, 4, int(im_size/2), int(im_size/2), dropout = dropout, padding='valid', strides=(1, 1))

    x = Lambda(flatten)(x)
    x = Activation('softmax')(x)

    model = Model(img_input, x)
    Adam = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss="categorical_crossentropy", optimizer=Adam, metrics=['accuracy'])
    return model


# Smaller  one with no downsampling at end
def conv_incp4(learning_rate = .001, dropout = .1, im_size=32):
    input_shape = (None, None, 3)
    img_input = Input(shape=input_shape)

    # Remove stride in the first layer, no max pool, 3 instead of 5
    x = conv2d_bn(img_input, 32, 3, 3, dropout = dropout, padding='same', strides=(1, 1))
    x = conv2d_bn(x, 32, 3, 3, dropout = dropout, padding='same', strides=(1, 1))

    # Inception Blocks - Mixed 1
    branch1x1 = conv2d_bn(x, 32, 1, 1, dropout = dropout)

    branch5x5 = conv2d_bn(x, 24, 1, 1, dropout = dropout)
    branch5x5 = conv2d_bn(branch5x5, 32, 5, 5, dropout = dropout)

    branch3x3dbl = conv2d_bn(x, 32, 1, 1, dropout = dropout)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 48, 3, 3, dropout = dropout)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 48, 3, 3, dropout = dropout)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 16, 1, 1, dropout = dropout)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool])

    # mixed 2
    branch1x1 = conv2d_bn(x, 16, 1, 1, dropout = dropout)

    branch5x5 = conv2d_bn(x, 12, 1, 1, dropout = dropout)
    branch5x5 = conv2d_bn(branch5x5, 32, 5, 5, dropout = dropout)

    branch3x3dbl = conv2d_bn(x, 16, 1, 1, dropout = dropout)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 24, 3, 3, dropout = dropout)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 24, 3, 3, dropout = dropout)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 16, 1, 1, dropout = dropout)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool])

    x = conv2d_bn(img_input, 16, 3, 3, dropout = dropout, padding='same', strides=(2, 2))
    x = conv2d_bn(x, 4, int(im_size/2), int(im_size/2), dropout = dropout, padding='valid', strides=(1, 1))

    x = Lambda(flatten)(x)
    x = Activation('softmax')(x)

    model = Model(img_input, x)
    Adam = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss="categorical_crossentropy", optimizer=Adam, metrics=['accuracy'])
    return model


# Incep3 was best, so try one with only one inception block
def conv_incp5(learning_rate = .001, dropout = .1, im_size=32):
    input_shape = (None, None, 3)
    img_input = Input(shape=input_shape)

    # Remove stride in the first layer, no max pool, 3 instead of 5
    x = conv2d_bn(img_input, 32, 3, 3, dropout = dropout, padding='same', strides=(1, 1))
    x = conv2d_bn(img_input, 32, 3, 3, dropout = dropout, padding='same', strides=(1, 1))
    x = conv2d_bn(x, 32, 3, 3, dropout = dropout, padding='same', strides=(2, 2))

    # Inception Blocks - Mixed 1
    branch1x1 = conv2d_bn(x, 32, 1, 1, dropout = dropout)

    branch5x5 = conv2d_bn(x, 24, 1, 1, dropout = dropout)
    branch5x5 = conv2d_bn(branch5x5, 32, 5, 5, dropout = dropout)

    branch3x3dbl = conv2d_bn(x, 32, 1, 1, dropout = dropout)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 48, 3, 3, dropout = dropout)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 48, 3, 3, dropout = dropout)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 16, 1, 1, dropout = dropout)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool])

    x = conv2d_bn(x, 4, int(im_size/2), int(im_size/2), dropout = dropout, padding='valid', strides=(1, 1))

    x = Lambda(flatten)(x)
    x = Activation('softmax')(x)

    model = Model(img_input, x)
    Adam = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss="categorical_crossentropy", optimizer=Adam, metrics=['accuracy'])
    return model
