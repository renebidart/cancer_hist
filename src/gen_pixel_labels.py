import os
import sys
import numpy as np
import pandas as pd
import random
import glob
import scipy.misc
import matplotlib.pyplot as plt 
from matplotlib.pyplot import imshow
from PIL import Image
from bs4 import BeautifulSoup
import xml.etree.cElementTree as ET
from numpy import linalg as LA
import h5py

from functions import*

def main(data_loc, out_dir, radius=10):
    radius=int(radius)

    # Folders  had better be labelled properly
    train_dir_in=os.path.join(data_loc, "train")
    valid_dir_in=os.path.join(data_loc, 'valid')
    test_dir_in=os.path.join(data_loc, 'test')

    # make the folders for output  
    train_dir_out=os.path.join(out_dir, "train")
    valid_dir_out=os.path.join(out_dir, 'valid')
    test_dir_out=os.path.join(out_dir, 'test')

    if not os.path.exists(train_dir_out):
        os.makedirs(train_dir_out)
    if not os.path.exists(valid_dir_out):
        os.makedirs(valid_dir_out)
    if not os.path.exists(test_dir_out):
        os.makedirs(test_dir_out)

    def add_nuclei(image, row_loc, col_loc, radius, type):
        for row in range(-1*radius, radius, 1):
            for col in range(-1*radius, radius, 1):
                # dont't just do a square
                dist = np.sqrt(row** 2 + col** 2)
                if (dist<=radius):
                    image[int(row_loc+row), int(col_loc+col), 0] = 0
                    image[int(row_loc+row), int(col_loc+col), type] = 1
        return image

    def gen_label_img(data_loc, out_dir, radius):
        # output is a 4 class image corresponding to 
        #.the radius al 
        all_files=glob.glob(os.path.join(data_loc, '*'))
        all_images = [loc for loc in all_files if loc.rsplit('.', 1)[-2][-4:] == 'crop']
        
        folder_size = len(all_images)
        print('folder_size: ', folder_size)
        
        for image_file in all_images:
          
            image = np.array(Image.open(image_file))
            im_name = str(image_file.rsplit('.', 1)[-2].rsplit('/', 1)[1].rsplit('_', 1)[0].replace(" ", "_"))

            label_image = np.zeros((np.shape(image)[0]+2*radius, np.shape(image)[1]+2*radius, 4)) # same size as the image, but will use 4 classes

            label_image[:,:, 0] = 1 # first set the entrie image to non-cell

            xml_loc = image_file.rsplit('_', 1)[0]+'_key.xml'

            # now read in the xml file, and add create a 4 channel image corresponding to the nuclei class
            point_list = get_points_xml(xml_loc, verbose=0)

            for index in range(np.shape(point_list)[0]):
                row_loc = point_list[index, 0]
                col_loc = point_list[index, 1]
                nucleus_type = point_list[index, 2]

                label_image = add_nuclei(image=label_image, row_loc=row_loc, col_loc=col_loc, radius=radius, type=int(nucleus_type))

            # remove the unnecessary padding
            label_image = label_image[radius:np.shape(label_image)[0]-radius, radius:np.shape(label_image)[1]-radius, :]
            
            # save the output as npy
            outfile=os.path.join(out_dir, im_name)
            np.save(outfile, label_image)

    gen_label_img(data_loc=train_dir_in, out_dir=train_dir_out, radius=radius)
    gen_label_img(data_loc=valid_dir_in, out_dir=valid_dir_out, radius=radius)
    gen_label_img(data_loc=test_dir_in, out_dir=test_dir_out, radius=radius)

if __name__ == "__main__":
    data_loc = sys.argv[1]
    out_dir = sys.argv[2]
    radius = sys.argv[3]

    main(data_loc, out_dir, radius)