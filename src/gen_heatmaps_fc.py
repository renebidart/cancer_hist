from __future__ import absolute_import, division, print_function, unicode_literals
import os
import sys
sys.path.insert(1, '/home/rbbidart/.local-mosaic/lib/python2.7/site-packages')

import numpy as np
import pandas as pd
import random
import glob
from PIL import Image
from keras.models import load_model
from keras.models import Model



def main(data_loc, out_dir, model_loc, height = 64, downsample = 2):

    def create_heatmap(image_loc, model_loc, height, downsample):
        def softmax(x):
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()
        
        image = np.array(Image.open(image_loc))
        image_shape = image.shape
        image = image/255.0 # During training the images were normalized
        height = int(height)
        
        model = load_model(model_loc)
        last = model.layers[-2].output
        model = Model(model.input, last)

        out_shape = np.ceil(np.array(image.shape)/float(downsample)).astype(int)
        out_shape[2] = 4 # there are 4 classes

        delta=int((height)/2)
        image = np.lib.pad(image, ((delta, delta-int(downsample)), (delta, delta-int(downsample)), (0,0)), 'constant', constant_values=(0, 0))
        image = np.expand_dims(image, axis=0)
        heat = model.predict(image, batch_size=1, verbose=0)
        heat = np.reshape(heat, out_shape)
        return heat

    model = load_model(model_loc)
    # Make the output folders
    dir_list = ['valid', 'test', 'train']

    for folder in dir_list:
        curr_folder = os.path.join(data_loc, folder)
        print('curr_folder: ', curr_folder)
        curr_out_folder = os.path.join(out_dir, folder)
        if not os.path.exists(curr_out_folder):
            os.makedirs(curr_out_folder)

        all_files=glob.glob(os.path.join(curr_folder, '*'))
        all_images = [loc for loc in all_files if loc.rsplit('.', 1)[-2][-4:] == 'crop']

        for image_loc in all_images:
            heat = create_heatmap(image_loc=image_loc, model_loc=model_loc, height=height, downsample=downsample)

            im_name = image_loc.rsplit('.', 1)[-2].rsplit('/', 1)[1]
            outfile=os.path.join(curr_out_folder, im_name)
            np.save(outfile, heat)

        # Add the xml ones too
        all_xml = [loc for loc in all_files if loc.rsplit('.', 1)[-2][-4:] == 'key']
        for file in all_xml:
            name = file.rsplit('/', 1)[1].replace(" ", "_")
            new_loc=os.path.join(curr_out_folder, name)
            copyfile(file, new_loc)


if __name__ == "__main__":
    data_loc = sys.argv[1]
    out_dir = sys.argv[2]
    model_loc = sys.argv[3]
    height = sys.argv[4]
    downsample = sys.argv[5]

    main(data_loc, out_dir, model_loc, height, downsample)