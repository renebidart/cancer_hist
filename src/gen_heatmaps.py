import os
import sys
sys.path.insert(1, '/home/rbbidart/.local-mosaic/lib/python2.7/site-packages')

import numpy as np
import pandas as pd
import random
import glob
from PIL import Image
from keras.models import load_model


def main(data_loc, out_dir, model_loc, stride=2, height = 32):

    def create_heatmap(image_loc, model, stride, height):
        image = np.array(Image.open(image_loc))
        image = image/255.0 # During training the images were normalized
        height = int(height)
        stride = int(stride)
        
        out_size = (np.floor(np.array(image.shape)/stride)).astype(int)
        out_size[2] = 4 # there are 4 classes
        out_image = np.zeros(out_size.astype(int))

        delta=int((height)/2)
        image = np.lib.pad(image, ((delta, delta), (delta, delta), (0,0)), 'constant', constant_values=(0, 0))

        # Double for loop is hideous and slow. Probably should be done in GPU somehow.
        for row in range(0, out_image.shape[0], 1):
            for col in range(0, out_image.shape[1], 1):
                in_col = int(col*stride+delta)
                in_row = int(row*stride+delta)
                seg_image = image[in_row-delta:in_row+delta, in_col-delta:in_col+delta,:]
                seg_image = np.expand_dims(seg_image, axis=0) # keras expects batchsize as index 0
                pred = model.predict(seg_image, batch_size=1, verbose=0)
                out_image[row, col, :] = pred
        return out_image

    model = load_model(model_loc)

    # Make the output folders
    dir_list = ['test', 'valid', 'train']

    for folder in dir_list:
        curr_folder = os.path.join(data_loc, folder)
        print curr_folder
        curr_out_folder = os.path.join(out_dir, folder)
        if not os.path.exists(curr_out_folder):
            os.makedirs(curr_out_folder)

        all_files=glob.glob(os.path.join(curr_folder, '*'))
        all_images = [loc for loc in all_files if loc.rsplit('.', 1)[-2][-4:] == 'crop']

        for image_loc in all_images:
            heat = create_heatmap(image_loc=image_loc, model=model, stride=stride, height=height)

            im_name = image_loc.rsplit('.', 1)[-2].rsplit('/', 1)[1]
            print im_name
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
    stride = sys.argv[4]
    height = sys.argv[5]

    main(data_loc, out_dir, model_loc, stride=stride, height=height)