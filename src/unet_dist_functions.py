import os
import sys
import glob
import pandas
import numpy as np
import random
from shutil import copyfile

from PIL import Image
from bs4 import BeautifulSoup
import xml.etree.cElementTree as ET
from matplotlib.pyplot import imshow



def create_heatmap(image_loc, model, height, downsample):
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    image = np.asarray(Image.open(image_loc))
    image_shape = image.shape
    image = image/255.0 # During training the images were normalized

    last = model.layers[-2].output
    model = Model(model.input, last)

    out_shape = np.ceil(np.array(image.shape)/float(downsample)).astype(int)
    out_shape[2] = 2 # there are 2 classes

    delta=int((height)/2)
    image = np.lib.pad(image, ((delta, delta-int(downsample)), (delta, delta-int(downsample)), (0,0)), 'constant', constant_values=(0, 0))
    image = np.expand_dims(image, axis=0)
    heat = model.predict(image, batch_size=1, verbose=0)
    heat = np.reshape(heat, out_shape)
    # now apply the softmax to only the 3 classes(not the overall class probability (why??))
    heat[:,:,:] = np.apply_along_axis(softmax, 2, heat[:,:,:])
    return heat[:,:,1]


def get_points_from_xml(xml_file):
    lymphocyte=['TIL-E', 'TIL-S']
    normal_epithelial=['normal', 'UDH', 'ADH']
    malignant_epithelial=['IDC', 'ILC', 'MucC', 'DCIS1', 'DCIS2', 'DCIS3', 'MC-E', 'MC-C', 'MC-M']
    
    with open(xml_file) as fp:
        soup = BeautifulSoup(fp, 'xml')
    groups=soup.find_all('graphic')

    num_pos = 0
    all_points=[]
    for group in groups:
        points=group.find_all('point')

        nucleus_type = group.get('description').replace(" ", "")
        if (nucleus_type in lymphocyte):
            label = '1'
        elif (nucleus_type in normal_epithelial):
            label = '2'
        elif (nucleus_type in malignant_epithelial):
            label = '3'
        else:
            # convention is to use the last valid label, meaning we shouldn't change the label variable 
            try:
                label
            except NameError:
                print("Error, no matching label with no prev obs - set var to 3")
                print('nucleus_type is: ', nucleus_type)
                print('File is ', xml_file)
                label = 3
            else:
                print ("Error, set var to prev obs: ", label)
                print ('nucleus_type is: ', nucleus_type)
                print ('File is ', xml_file)

        for point in points:
            x=int(point.get_text().rsplit(',', 1)[0])
            y=int(point.get_text().rsplit(',', 1)[1])
            all_points.append([x,y, label])
    all_points = np.array(all_points).astype(float)
    return all_points

def make_sample_unet_dataset(data_loc, out_loc, downsample_factor):
    data_classes = glob.glob(data_loc+'/*')
    for data_class_loc in data_classes:
        imgs = glob.glob(data_class_loc+'/*')
        imgs = [loc for loc in imgs if loc.rsplit('.', 1)[-1] in ['tif']]
        data_class = data_class_loc.rsplit('/', 1)[1]
        num = len(imgs)
        sample_num = int(num/downsample_factor)
        samp = np.random.choice(imgs, sample_num)
        print('Total number: ', num, 'Sample number: ', len(samp))
        for file in samp:
            name = file.rsplit('/', 1)[1]
            new_loc = os.path.join(out_loc, data_class)
            if not os.path.exists(new_loc):
                os.makedirs(new_loc)
            copyfile(file, os.path.join(new_loc , name))



def plot_heatmap_simple(img_loc, model):
	# look at the heatmaps produced, and see how they compare to the actual image
	# only 3 of the 4 heatmaps are shown
    f = plt.figure(figsize=(15, 15))
    for img_num, img_locs in enumerate(image_locs):
        xml_loc = img_locs.replace('crop.tif', 'key.xml')
        image = np.asarray(Image.open(img_locs))
        
        image_pad = np.zeros((512, 1024, 3))
        image_pad[:image.shape[0],:image.shape[1] , :] = image
        image_pad = np.expand_dims(image_pad, axis=0)

        heat = model.predict(image_pad, batch_size=1, verbose=0)
        heat = np.squeeze(heat)
        heat = heat[:image.shape[0],:image.shape[1] , :]

        true_pts = get_points_xml(xml_loc)

        sp = f.add_subplot(2, 4//2, 1)
        sp.axis('Off')
        sp.set_title('Raw Image with Nuclei', fontsize=20)
        image2 = np.array(image)
        for row in range(len(true_pts)):
            if preds[row, 2] == 1:
                color = [0, 0, 255]
            elif preds[row, 2] == 2:
                color = [0, 255, 0]  
            elif preds[row, 2] == 3:
                color = [255, 0, 0]
            image2[int(true_pts[row, 0])-2:int(true_pts[row, 0])+2, int(true_pts[row, 1])-2:int(true_pts[row, 1])+2, :] = color
        plt.tight_layout()
        plt.imshow(image2)

        sp = f.add_subplot(2, 4//2, 2)
        sp.axis('Off')
        sp.set_title('Heatmap - channel 0', fontsize=20)
        plt.tight_layout()
        plt.imshow(heat[:, :, 0])

        sp = f.add_subplot(2, 4//2, 3)
        sp.axis('Off')
        sp.set_title('Heatmap - channel 1', fontsize=20)
        plt.tight_layout()
        plt.imshow(heat[:, :, 1])

        sp = f.add_subplot(2, 4//2, 4)
        sp.axis('Off')
        sp.set_title('Heatmap - channel 2', fontsize=20)
        plt.tight_layout()
        plt.imshow(heat[:, :, 2])
