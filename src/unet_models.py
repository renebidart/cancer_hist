import os
import numpy as np
import glob
from scipy.ndimage import rotate
from PIL import Image

import keras
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
from keras import metrics
from keras import layers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Reshape, Input, concatenate, Conv2DTranspose
from keras.layers.core import Activation, Dense, Lambda
from keras.constraints import maxnorm
from keras.optimizers import SGD, Adam
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization


############ DATA GENERATORS

def data_gen_aug_combined(file_loc, mask_loc, batch_size, square_rot_p=.3, seed=101):
    # square_rot_p is the prob of using a 90x rotation, otherwise sample from 360. Possibly not useful
    # translate is maximum number of pixels to translate by. Make it close the doctor's variance in annotation
    square_rot_p = int(square_rot_p)
    np.random.seed(seed)
    all_files=glob.glob(os.path.join(file_loc, '*'))
    all_masks=[]

    all_files = [loc for loc in all_files if loc.rsplit('.', 1)[-1] in ['tif']]

    for file in all_files:
        im_name = str(file.rsplit('.', 1)[-2].rsplit('/', 1)[1].rsplit('_', 1)[0].replace(" ", "_"))
        loc = os.path.join(mask_loc, im_name+'.npy')
        all_masks.append(loc)

    while 1:
        c = list(zip(all_files, all_masks))
        np.random.shuffle(c)
        all_files, all_masks = zip(*c)

        num_batches = int(np.floor(len(all_files)/batch_size))-1

        for batch in range(num_batches):
            x=[]
            y=[]
            batch_files = all_files[batch_size*batch:batch_size*(batch+1)]
            batch_files_mask = all_masks[batch_size*batch:batch_size*(batch+1)]

            for index in range(len(batch_files)):
                image_loc = batch_files[index]
                mask_loc = batch_files_mask[index]

                # load the image
                image = Image.open(image_loc)

                width, height = image.size
                image = np.reshape(np.array(image.getdata()), (height, width, 3))

                #load the mask
                mask = np.load(mask_loc)

                # All the randomness:
                height, width = np.shape(image)[0], np.shape(image)[1]
                crop_row = np.random.randint(0, height-320)
                crop_col = np.random.randint(0, width-368)
                flip_vert = np.random.randint(0, 2)
                flip_hor = np.random.randint(0, 2)

                # APPLY AUGMENTATION:
                # flips
                if flip_vert:
                    image = np.flipud(image)
                    mask = np.flipud(mask)

                if flip_hor:
                    image = np.fliplr(image)
                    mask = np.fliplr(mask)

                # rotation
                square_rot =  bool((np.random.uniform(0, 1, 1)<square_rot_p))
                if square_rot:  # maybe this is dumb, but it cant hurt
                    rotations=['0', '90', '180', '270']
                    angle = int(random.choice(rotations))
                    image = rotate(image, angle, reshape=False)
                    mask = rotate(mask, angle, reshape=False)

                else:
                    angle = np.random.uniform(0, 368,1)
                    image = rotate(image, angle, reshape=False)
                    mask = rotate(mask, angle, reshape=False)
 
                # crop to 320 x 360 so it will fit into network, and for data augmentation
                image = image[crop_row:crop_row+320, crop_col:crop_col+368]
                mask = mask[crop_row:crop_row+320, crop_col:crop_col+368]

                image = image/255.0 # make pixels in [0,1] 
                x.append(image)
                y.append(mask)

            x=np.array(x)
            y=np.array(y)
            yield (x, y)

def data_gen_combined(file_loc, mask_loc, batch_size, seed=101):
    # square_rot_p is the prob of using a 90x rotation, otherwise sample from 360. Possibly not useful
    # translate is maximum number of pixels to translate by. Make it close the doctor's variance in annotation
    np.random.seed(seed)
    all_files=glob.glob(os.path.join(file_loc, '*'))
    all_masks=glob.glob(os.path.join(mask_loc, '*'))

    all_files = [loc for loc in all_files if loc.rsplit('.', 1)[-1] in ['tif']]

    while 1:
        c = list(zip(all_files, all_masks))
        np.random.shuffle(c)

        all_files, all_masks = zip(*c)

        num_batches = int(np.floor(len(all_files)/batch_size))-1
        for batch in range(num_batches):
            x=[]
            y=[]
            batch_files = all_files[batch_size*batch:batch_size*(batch+1)]
            batch_files_mask = all_masks[batch_size*batch:batch_size*(batch+1)]

            for index in range(len(batch_files)):
                image_loc = batch_files[index]
                mask_loc = batch_files_mask[index]

                # load the image
                image = Image.open(image_loc)
                width, height = image.size
                image = np.reshape(np.array(image.getdata()), (height, width, 3))

                #load the mask
                mask = np.load(mask_loc)

                # make it the same size as the training examples
                height, width = np.shape(image)[0], np.shape(image)[1]
                crop_row = np.random.randint(0, height-320)
                crop_col = np.random.randint(0, width-368)

                # crop to 320 x 360 so it will fit into network, and for data augmentation
                image = image[crop_row:crop_row+320, crop_col:crop_col+368]
                mask = mask[crop_row:crop_row+320, crop_col:crop_col+368]

                image = image/255.0 # make pixels in [0,1]     
                x.append(image)
                y.append(mask)

            x=np.array(x)
            y=np.array(y)
            yield (x, y)



############ PIXEL LOSS FUNCTION
# from https://github.com/jocicmarko/ultrasound-nerve-segmentation
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

smooth = 1. # should this be adjusted?


# Use a multiclass unet-style architecture on the fake labelled data

def conv_block(x,
              filters,
              num_row,
              num_col,
              dropout, 
              padding='same',
              strides=(1, 1),
              activation='relu'):
    x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, activation=activation)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout)(x)

    x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, activation=activation)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout)(x)
    return x



############ UNET ARCHITECTURES 
def unet_dp_bn(learning_rate=.0001):
    input_shape = (None, None, 3)
    img_input = Input(shape=input_shape)

    conv1 = conv_block(img_input, 32, 3, 3, dropout = .1, padding='same', strides=(1, 1), activation='relu')
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = conv_block(pool1, 64, 3, 3, dropout = .1, padding='same', strides=(1, 1), activation='relu')
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = conv_block(pool2, 128, 3, 3, dropout = .1, padding='same', strides=(1, 1), activation='relu')
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = conv_block(pool3, 256, 3, 3, dropout = .1, padding='same', strides=(1, 1), activation='relu')
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = conv_block(pool4, 512, 3, 3, dropout = .1, padding='same', strides=(1, 1), activation='relu')

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = conv_block(up6, 256, 3, 3, dropout = .1, padding='same', strides=(1, 1), activation='relu')

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = conv_block(up7, 128, 3, 3, dropout = .1, padding='same', strides=(1, 1), activation='relu')

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = conv_block(up8, 64, 3, 3, dropout = .1, padding='same', strides=(1, 1), activation='relu')

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = conv_block(up9, 32, 3, 3, dropout = .1, padding='same', strides=(1, 1), activation='relu')

    conv10 = Conv2D(4, (1, 1), activation='softmax')(conv9)

    model = Model(img_input, conv10)
    model.compile(optimizer=Adam(lr=learning_rate), loss=dice_coef_loss, metrics=[dice_coef])
    return model


def unet_standard(learning_rate=.0001):
    input_shape = (None, None, 3)
    img_input = Input(shape=input_shape)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(img_input)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(4, (1, 1), activation='softmax')(conv9)

    model = Model(img_input, conv10)
    model.compile(optimizer=Adam(lr=learning_rate), loss=dice_coef_loss, metrics=[dice_coef])
    return model



def half_n_half(learning_rate=.0001):
    input_shape = (None, None, 3)
    img_input = Input(shape=input_shape)

    conv1 = conv_block(img_input, 32, 3, 3, dropout = .1, padding='same', strides=(1, 1), activation='relu')
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = conv_block(pool1, 64, 3, 3, dropout = .1, padding='same', strides=(1, 1), activation='relu')
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = conv_block(pool2, 64, 3, 3, dropout = .1, padding='same', strides=(1, 1), activation='relu')
    conv4 = conv_block(conv3, 64, 3, 3, dropout = .1, padding='same', strides=(1, 1), activation='relu')
    conv5 = conv_block(conv4, 64, 3, 3, dropout = .1, padding='same', strides=(1, 1), activation='relu')

    up6 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv5), conv2], axis=3)
    conv6 = conv_block(up6, 64, 3, 3, dropout = .1, padding='same', strides=(1, 1), activation='relu')

    up7 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv6), conv1], axis=3)
    conv7 = conv_block(up7, 32, 3, 3, dropout = .1, padding='same', strides=(1, 1), activation='relu')

    conv10 = Conv2D(4, (1, 1), activation='softmax')(conv7)

    model = Model(img_input, conv10)
    model.compile(optimizer=Adam(lr=learning_rate), loss=dice_coef_loss, metrics=[dice_coef])
    return model


def unet_paper(learning_rate=.0001):
    input_shape = (None, None, 3)
    img_input = Input(shape=input_shape)

    conv1 = conv_block(img_input, 32, 3, 3, dropout = .1, padding='same', strides=(1, 1), activation='relu')
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = conv_block(pool1, 64, 3, 3, dropout = .1, padding='same', strides=(1, 1), activation='relu')
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = conv_block(pool2, 128, 3, 3, dropout = .1, padding='same', strides=(1, 1), activation='relu')
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = conv_block(pool3, 128, 3, 3, dropout = .1, padding='same', strides=(1, 1), activation='relu')

    up5 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv4), conv3], axis=3)
    conv5 = conv_block(up5, 128, 3, 3, dropout = .1, padding='same', strides=(1, 1), activation='relu')

    up6 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv5), conv2], axis=3)
    conv6 = conv_block(up6, 64, 3, 3, dropout = .1, padding='same', strides=(1, 1), activation='relu')

    up7 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv6), conv1], axis=3)
    conv7 = conv_block(up7, 32, 3, 3, dropout = .1, padding='same', strides=(1, 1), activation='relu')
    
    conv8_dist = Conv2D(1, (1, 1), activation='sigmoid')(conv7)
    conv8_cross_entropy = Conv2D(3, (1, 1), activation='softmax')(conv7)
    output = concatenate([conv8_dist, conv8_cross_entropy])
    
    model = Model(img_input, output)
    model.compile(optimizer=Adam(lr=learning_rate), loss=distance_loss, metrics=[distance_loss])
    return model


def distance_loss(y_true, y_pred):
    weight = .5 # how mush does the distance matter compared to the cross entropy (fast ai used .001 for 4 more uncertain ones)
    distance_loss = K.binary_crossentropy(y_pred[:, :, :, 0], y_true[:, :, :, 0])    
    cross_entropy = K.categorical_crossentropy(y_true[:, :, :, 1:], y_pred[:, :, :, 1:])    

    return(distance_loss*weight+(1-weight)*cross_entropy)


