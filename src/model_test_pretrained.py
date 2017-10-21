from __future__ import absolute_import, division, print_function, unicode_literals

from heat_models import*
import sys
sys.path.insert(1, '/home/rbbidart/.local-mosaic/lib/python2.7/site-packages')


def main(data_loc, out_loc, model_loc, epochs, batch_size, im_size, model_str): 
    import os
    import glob
    import random
    import numpy as np 
    import pandas as pd
    import keras
    import pickle
    from keras import backend as K
    from keras.engine.topology import Layer
    from keras.layers import Dropout, Flatten, Reshape, Input
    from keras.layers.core import Activation, Dense, Lambda
    from keras.callbacks import ModelCheckpoint, EarlyStopping
    from keras.models import load_model


    # Get the function:
    functionList = {
    'conv1': conv1,
    'conv_fc2': conv_fc2,
    'conv_fc3': conv_fc3,
    'conv_fc8': conv_fc8,
    'conv_fc7': conv_fc7,
    'conv_incp3' : conv_incp3
    }

    parameters = {
    'dropout': .1,
    'im_size': int(im_size)
    }

    # Locations
    train_loc = os.path.join(str(data_loc),'train')
    valid_loc = os.path.join(str(data_loc),'valid')
    num_train = len(glob.glob(os.path.join(train_loc, '*')))
    num_valid = len(glob.glob(os.path.join(valid_loc, '*')))
    print(valid_loc)
    print('num_train', num_train)
    print('num_valid', num_valid)


    def triangular2(epoch): 
        base_lr = .0002
        max_lr = .002
        step_size = 5
        cycle = np.floor(1+epoch/(2*step_size))
        x = np.abs(epoch/step_size - 2*cycle + 1)
        lr = base_lr + (max_lr-base_lr)*np.maximum(0, (1-x))/float(2**(cycle-1))
        return lr

    # Params for all models
    epochs=int(epochs)
    batch_size=int(batch_size)   # make this divisible by len(x_data)
    steps_per_epoch = np.floor(num_train/batch_size) # num of batches from generator at each epoch. (make it full train set)
    validation_steps = np.floor(num_valid/batch_size) # size of validation dataset divided by batch size
    print('validation_steps', validation_steps)
    image_shape = (int(im_size), int(im_size))
    
    # Create the model and load its weights:
    model = functionList[model_str](**parameters)
    model.load_weights(model_loc)

    print(model.summary())
    name = model_str+'_'+im_size
    out_file=os.path.join(str(out_loc), name)
    checkpointer = ModelCheckpoint(filepath=os.path.join(out_loc, name+'_.{epoch:02d}-{val_acc:.2f}.hdf5'), verbose=1, monitor='val_loss', save_best_only=True)

    callbacks = [EarlyStopping(monitor='val_loss', patience=15, verbose=0),
        keras.callbacks.LearningRateScheduler(triangular2),
        ModelCheckpoint(filepath=os.path.join(out_loc, name + '_.{epoch:02d}-{val_acc:.2f}.hdf5'), 
            verbose=1, monitor='val_loss', save_best_only=True)]

    hist = model.fit_generator(data_gen(train_loc, batch_size, im_size=im_size, 
                                      square_rot_p=.5, translate=0, flips=True, rotate=True),
                                      validation_data=data_gen(train_loc, batch_size, im_size=im_size),
                                      steps_per_epoch=steps_per_epoch, 
                                      epochs=epochs,
                                      validation_steps=validation_steps,
                                      callbacks=callbacks)
    pickle.dump(hist.history, open(out_file, 'wb'))


if __name__ == "__main__":
    data_loc = sys.argv[1]
    out_loc = sys.argv[2]
    epochs = sys.argv[3]
    batch_size = sys.argv[4]
    im_size = sys.argv[5]
    model_str = sys.argv[6]

    main(data_loc, out_loc, epochs, batch_size, im_size, model_str)