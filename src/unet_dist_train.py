from unet_dist_models import*
import sys


def main(data_loc, mask_loc, out_loc, epochs, batch_size, model_str, d_weight):
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

    
    d_weight = float(d_weight)
    def distance_loss(y_true, y_pred):
        weight = d_weight # how mush does the distance matter compared to the cross entropy (fast ai used .001 for 4 more uncertain ones)
        distance_loss = K.binary_crossentropy(y_pred[:, :, :, 0], y_true[:, :, :, 0])    
        cross_entropy = K.categorical_crossentropy(y_true[:, :, :, 1:], y_pred[:, :, :, 1:])    

        return(distance_loss*weight+(1-weight)*cross_entropy)

    # Get the model:
    functionList = {
    'unet_standard': unet_standard,
    'unet_mid': unet_mid,
    'unet_mid2': unet_mid2,
    'unet_paper': unet_paper
    }

    parameters = {
    'learning_rate': .0001    
    }

    epochs=int(epochs)
    batch_size=int(batch_size)


    # Locations
    if not os.path.exists(out_loc):
        os.makedirs(out_loc)    
    train_loc = os.path.join(str(data_loc),'train', str(0))
    train_mask_loc = os.path.join(str(mask_loc),'train', str(0))

    valid_loc = os.path.join(str(data_loc),'valid', str(0))
    valid_mask_loc = os.path.join(str(mask_loc),'valid', str(0))

    num_train = len(glob.glob(os.path.join(train_loc, '*')))
    num_valid = len(glob.glob(os.path.join(valid_loc, '*')))
    print(valid_loc)
    print('num_train', num_train)
    print('num_valid', num_valid)

    # Params for all models
    batch_size=int(batch_size)   # make this divisible by len(x_data)
    steps_per_epoch = np.floor(num_train/batch_size) # num of batches from generator at each epoch. (make it full train set)
    validation_steps = np.floor(num_valid/batch_size)# size of validation dataset divided by batch size
    print('validation_steps', validation_steps)

    # need a batch generator to augment the labels same as the train images
    valid_generator = data_gen_combined(valid_loc, valid_mask_loc, batch_size, seed=101)
    train_generator = data_gen_aug_combined(train_loc, train_mask_loc, batch_size, square_rot_p=.3,  seed=101)

    model = functionList[model_str](**parameters)
    print(model.summary())
    name = model_str+'_'+str(d_weight)
    out_file=os.path.join(str(out_loc), name)

    callbacks = [EarlyStopping(monitor='distance_loss', patience=15, verbose=1),
        ModelCheckpoint(filepath=os.path.join(out_loc, name + '_.{epoch:02d}-{distance_loss:.2f}.hdf5'), 
            verbose=1, monitor='distance_loss', save_best_only=True)]

    hist = model.fit_generator(train_generator,
                                      validation_data=valid_generator,
                                      steps_per_epoch=steps_per_epoch, 
                                      epochs=epochs,
                                      validation_steps=validation_steps,
                                      callbacks=callbacks)
    pickle.dump(hist.history, open(out_file, 'wb'))

if __name__ == "__main__":
    data_loc = sys.argv[1]
    mask_loc = sys.argv[2]
    epochs = sys.argv[3]
    batch_size = sys.argv[4]
    model_str = sys.argv[5]
    d_weight = sys.argv[6]

    main(data_loc, mask_loc, epochs, batch_size, model_str, d_weight)
