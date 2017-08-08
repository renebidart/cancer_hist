# Way too much repeated code, but I don't want to change the names of anything or variables for reproduceability.

import os
import sys
sys.path.insert(1, '/home/rbbidart/.local-mosaic/lib/python2.7/site-packages')
import numpy as np
import pandas as pd
import random
import glob
import random
import scipy
import scipy.ndimage
from PIL import Image
from bs4 import BeautifulSoup


import keras
from keras import layers
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Reshape, Input
from keras.layers.core import Activation, Dense, Lambda
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization

# Make generator take as input a random segment of size 100x100 from the image
# Does it matter if all the images in one batch come from the same image?

def data_gen_loc_aug(file_loc, batch_size, image_shape=(100, 100)):
    # the number of images to take the samples from during each batch. 
    # Increasing batch size will cause more samples from each image, it will still take num_img images per batch
    # A different way is fixing the number of samples from each image, and making the batch take from more images
    num_img = 4
    bad_files = []
    while 1:
        all_files=glob.glob(os.path.join(file_loc, '*'))
        all_files = [loc for loc in all_files if loc.rsplit('.', 1)[-2][-4:] == 'crop']
        random.shuffle(all_files) # randomize order after every epoch
        num_batches = int(np.floor(len(all_files)/num_img)) # we will always take num_img per batch
        
        for batch in range(num_batches):
            x=[]
            y=[]
            batch_files = all_files[num_img*batch:num_img*(batch+1)]

            for image_file in batch_files:
                x_temp, y_temp = get_imgs_noloc(image_file, num_regions = int(batch_size/num_img), im_size = image_shape[0])
                y.extend(y_temp)
                x.extend(x_temp)
            x=np.array(x)
            y=np.array(y)
            
            yield (x, y)

def data_gen_loc_no_aug(file_loc, batch_size, image_shape=(100, 100)):
    # the number of images to take the samples from during each batch. 
    # Increasing batch size will cause more samples from each image, it will still take num_img images per batch
    # A different way is fixing the number of samples from each image, and making the batch take from more images
    num_img = 4
    bad_files = []
    while 1:
        all_files=glob.glob(os.path.join(file_loc, '*'))
        all_files = [loc for loc in all_files if loc.rsplit('.', 1)[-2][-4:] == 'crop']
        random.shuffle(all_files) # randomize order after every epoch
        num_batches = int(np.floor(len(all_files)/num_img)) # we will always take num_img per batch
        
        for batch in range(num_batches):
            x=[]
            y=[]
            batch_files = all_files[num_img*batch:num_img*(batch+1)]

            for image_file in batch_files:
                x_temp, y_temp = get_imgs_noloc_aug(image_file, num_regions = int(batch_size/num_img), im_size = image_shape[0])
                y.extend(y_temp)
                x.extend(x_temp)
            x=np.array(x)
            y=np.array(y)
            
            yield (x, y)

def get_imgs_noloc(image_file, num_regions, im_size):
    im_size=int(im_size)
    x=[]
    y=[]
    full_image = np.array(Image.open(image_file))
    # pad the image so we can evenly sample from the area of the image
    pad_width=int((im_size)/2)
    full_image = np.lib.pad(full_image, ((pad_width, pad_width), (pad_width, pad_width), (0,0)), 'constant', constant_values=(0, 0))
    full_image = full_image/255.0 # make pixels in [0,1]

    xml_loc = image_file.rsplit('_', 1)[0]+'_key.xml'
    loc_dat = get_points_xml(xml_loc = xml_loc)
    # Adjust the locations because we have padded the image by pad_width
    loc_dat['col'] += pad_width
    loc_dat['row'] += pad_width

    for i in range(num_regions):
        y_temp=np.zeros(3)
        row = random.randint(0, full_image.shape[0]-im_size)
        col = random.randint(0, full_image.shape[1]-im_size)
        seg_image = full_image[row:row+im_size, col:col+im_size,:]
        
        points_in = loc_dat.loc[(loc_dat.row >=row) & (loc_dat.row <= row+im_size) & (loc_dat.col >=col) & (loc_dat.col <= col+im_size)]
        if len(points_in.index)==0:
            data = np.array([['label','col','row'],
                ['0', 0, 0],
                ['1', 0, 0],
                ['2', 0, 0]])
                
            points_in = pd.DataFrame(data=data[1:,:],
                  columns=data[0,:])

        points_in.label = points_in.label.astype(float)
        points_in = points_in.groupby('label').count().add_suffix('_Count').reset_index()
        points_in = points_in.set_index("label").reindex([0, 1, 2]).reset_index().fillna(value=0)
        
        y.append(np.array(points_in['col_Count']))
        x.append(seg_image)
    return x, y


def get_imgs_noloc_aug(image_file, num_regions, im_size):
    im_size=int(im_size)
    x=[]
    y=[]
    full_image = np.array(Image.open(image_file))
    # pad the image so we can evenly sample from the area of the image
    pad_width=int((im_size)/2)
    full_image = np.lib.pad(full_image, ((pad_width, pad_width), (pad_width, pad_width), (0,0)), 'constant', constant_values=(0, 0))
    full_image = full_image/255.0 # make pixels in [0,1]

    xml_loc = image_file.rsplit('_', 1)[0]+'_key.xml'
    loc_dat = get_points_xml(xml_loc = xml_loc)
    # Adjust the locations because we have padded the image by pad_width
    loc_dat['col'] += pad_width
    loc_dat['row'] += pad_width

    for i in range(num_regions):
        y_temp=np.zeros(3)
        row = random.randint(0, full_image.shape[0]-im_size)
        col = random.randint(0, full_image.shape[1]-im_size)
        seg_image = full_image[row:row+im_size, col:col+im_size,:]

        ##### Do augmentation on the segmented image

        # Flips:
        flip_vert = random.randint(0, 1)
        flip_hor =random.randint(0, 1)
        if flip_vert:
            seg_image = np.fliplr(seg_image)
        if flip_hor:
            seg_image = np.flipud(seg_image)

        # ROTATION
        rotations=['0', '90', '180', '270']
        angle = int(random.choice(rotations))
        seg_image = scipy.ndimage.rotate(seg_image, angle, reshape=False)
        
        points_in = loc_dat.loc[(loc_dat.row >=row) & (loc_dat.row <= row+im_size) & (loc_dat.col >=col) & (loc_dat.col <= col+im_size)]
        if len(points_in.index)==0:
            data = np.array([['label','col','row'],
                ['0', 0, 0],
                ['1', 0, 0],
                ['2', 0, 0]])
                
            points_in = pd.DataFrame(data=data[1:,:],
                  columns=data[0,:])

        points_in.label = points_in.label.astype(float)
        points_in = points_in.groupby('label').count().add_suffix('_Count').reset_index()
        points_in = points_in.set_index("label").reindex([0, 1, 2]).reset_index().fillna(value=0)
        
        y.append(np.array(points_in['col_Count']))
        x.append(seg_image)
    return x, y

def get_points_xml(xml_loc):
    lymphocyte=['TIL-E', 'TIL-S']
    normal_epithelial=['normal', 'UDH', 'ADH']
    malignant_epithelial=['IDC', 'ILC', 'Muc C', 'DCIS1', 'DCIS2', 'DCIS3', 'MC-E', 'MC-C', 'MC-M']
    
    loc_dat=pd.DataFrame()
    
    with open(xml_loc) as fp:
        soup = BeautifulSoup(fp, 'xml')
    groups=soup.find_all('graphic')

    for group in groups:
        points=group.find_all('point')
        nucleus_type = group.get('description').replace(" ", "")
        if (nucleus_type in lymphocyte):
            label = '0'
        elif (nucleus_type in normal_epithelial):
            label = '1'
        elif (nucleus_type in malignant_epithelial):
            label = '2'
        else: # convention is to use the last valid label, meaning we shouldn't change the label variable  
            #print "error"
            label=0

        for point in points:
            col = int(point.get_text().rsplit(',', 1)[0])
            row = int(point.get_text().rsplit(',', 1)[1])
            loc_dat = loc_dat.append({'col': col, 'row': row, 'label': label}, ignore_index=True)
    return loc_dat



########################################

def conv1(learning_rate = 0.0005, dropout = .1, im_size=100):
    model = Sequential()
    model.add(Conv2D(16, (3, 3), padding='same', input_shape=(im_size, im_size, 3), kernel_initializer='he_normal'))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(im_size/2, im_size/2, 16), kernel_initializer='he_normal'))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(im_size/4, im_size/4, 32), kernel_initializer='he_normal'))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(512, kernel_initializer="he_normal"))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout*4))

    model.add(Dense(3, kernel_initializer="he_normal"))

    Adam = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss="mean_squared_error", optimizer=Adam, metrics=['accuracy'])
    return model

def conv2(learning_rate = 0.0005,  dropout = .1, im_size=100):
    model = Sequential()
    model.add(Conv2D(16, (3, 3), padding='same', input_shape=(im_size, im_size, 3), kernel_initializer='he_normal'))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(Conv2D(16, (3, 3), padding='same', input_shape=(im_size, im_size, 16), kernel_initializer='he_normal'))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(im_size/2, im_size/2, 16), kernel_initializer='he_normal'))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(im_size/2, im_size/2, 32), kernel_initializer='he_normal'))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(im_size/4, im_size/4, 32), kernel_initializer='he_normal'))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(keras.layers.normalization.BatchNormalization())

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(im_size/4, im_size/4, 32), kernel_initializer='he_normal'))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(512, kernel_initializer="he_normal"))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout*4))

    model.add(Dense(64, kernel_initializer="he_normal"))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout*4))

    model.add(Dense(3, kernel_initializer="he_normal"))

    Adam = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss="mean_squared_error", optimizer=Adam, metrics=['accuracy'])
    return model

## INCEPTION MODELS
# Custom functions to use because keras flatten doesn't like variable sized input
def flatten(x):
    batch_size = K.shape(x)[0]
    x = K.reshape(x, (batch_size, -1)) 
    return x

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

def conv_incp1(learning_rate = .001, dropout = .1, im_size=100):
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

    x = conv2d_bn(x, 3, int(im_size/2), int(im_size/2), dropout = dropout, padding='valid', strides=(1, 1))
    x = Lambda(flatten)(x)

    model = Model(img_input, x)
    Adam = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss="categorical_crossentropy", optimizer=Adam, metrics=['accuracy'])
    return model
