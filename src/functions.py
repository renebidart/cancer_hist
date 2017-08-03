from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import glob

from sklearn.metrics import accuracy_score
from PIL import Image

from bs4 import BeautifulSoup
import matplotlib.pyplot as plt


# Given the xml location, return all the points and their classes
def get_points_xml(xml_loc, verbose=0):
    lymphocyte=['TIL-E', 'TIL-S']
    normal_epithelial=['normal', 'UDH', 'ADH']
    malignant_epithelial=['IDC', 'ILC', 'Muc C', 'DCIS1', 'DCIS2', 'DCIS3', 'MC-E', 'MC-C', 'MC-M']
    row_list=[]
    col_list=[]
    label_list=[]

    with open(xml_loc) as fp:
        soup = BeautifulSoup(fp, 'xml')
    groups=soup.find_all('graphic')

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
                if verbose:
                    print("Error, no matching label with no prev obs - set var to 3")
                    print('nucleus_type is: ', nucleus_type)
                    print('File is ', xml_loc)
                label = 3
            else:
                if verbose:
                    print("Error, set var to prev obs: ", label)
                    print('nucleus_type is: ', nucleus_type)
                    print('File is ', xml_loc)
            if verbose:
                print('error', xml_loc)
            
        for point in points:
            col = int(point.get_text().rsplit(',', 1)[0])
            row = int(point.get_text().rsplit(',', 1)[1])
            row_list.append(row)
            col_list.append(col)
            label_list.append(label)
            
    loc_dat = np.column_stack((np.array(row_list), np.array(col_list), np.array(label_list)))
    loc_dat = loc_dat.astype(float)
    return loc_dat

# given a heatmap, outputs all the coordinates and their class probabilities.
# class=True just gives a label, class=False gives probabilities
def non_max_supression(heatmap, radius=5, cutoff=.5, stride=2, output_class=True): 
    labels = []
    points = []    
    heatmap[:, :, 0] = 1-heatmap[:, :, 0]
    heatmap = np.lib.pad(heatmap, ((radius, radius), (radius, radius), (0,0)), 'constant', constant_values=(0, 0))

    curr_max = 1
    while (curr_max > float(cutoff)):
        max_coord = np.asarray(np.unravel_index(heatmap[:, :, 0].argmax(), heatmap[:, :, 0].shape))
        # Add the maximum cell probability to the coordinate
        if output_class: 
            max_coord = np.append(max_coord, np.argmax(heatmap[max_coord[0], max_coord[1], 1:])+1)
        elif not output_class:
            max_coord = np.append(max_coord, heatmap[max_coord[0], max_coord[1], :])
            
        # find the max set all classes within radius r to p = 0
        curr_max = heatmap[:, :, 0].max()
        for row in range(-1*radius, radius, 1):
            for col in range(-1*radius, radius, 1):
                # dont't just do a square
                dist = np.sqrt(row** 2 + col** 2)
                if (dist<=radius):
                    heatmap[int(max_coord[0]+row), int(max_coord[1]+col), :] = 0
        # adjust for the padding that was added
        max_coord[0] = max_coord[0] - radius
        max_coord[1] = max_coord[1] - radius
        
        points.append(max_coord)
    points = np.array(points)
    #points = points[points[:,2]!=0]
    
    points = points.astype(float)
    points[:,0] = points[:,0]*stride
    points[:,1] = points[:,1]*stride
    return points


def non_max_supression_variable(heatmap, radius=2, cutoff=.5, stride=2, output_class=True): 
    labels = []
    points = []    
    heatmap[:, :, 0] = 1-heatmap[:, :, 0]
    radius_m = max(radius)
    heatmap = np.lib.pad(heatmap, ((radius_m, radius_m), (radius_m, radius_m), (0,0)), 'constant', constant_values=(0, 0))

    curr_max = 1
    while (curr_max > float(cutoff)):
        max_coord = np.asarray(np.unravel_index(heatmap[:, :, 0].argmax(), heatmap[:, :, 0].shape))
        # Add the maximum cell probability to the coordinate
        if output_class: 
            max_coord = np.append(max_coord, np.argmax(heatmap[max_coord[0], max_coord[1], 1:])+1)
        elif not output_class:
            max_coord = np.append(max_coord, heatmap[max_coord[0], max_coord[1], :])

        point_class = np.argmax(heatmap[max_coord[0], max_coord[1], 1:])+1

        if (point_class == 1):
            radius_act = radius[0]
        elif (point_class == 2):
            radius_act = radius[1]
        elif (point_class == 3):
            radius_act = radius[2]
        else:
            print('error. the radius cant be found')
            
        # find the max set all classes within radius r to p = 0
        curr_max = heatmap[:, :, 0].max()
        for row in range(-1*radius_act, radius_act, 1):
            for col in range(-1*radius_act, radius_act, 1):
                # dont't just do a square
                dist = np.sqrt(row** 2 + col** 2)
                if (dist<=radius_act):
                    heatmap[int(max_coord[0]+row), int(max_coord[1]+col), :] = 0
        # adjust for the padding that was added
        max_coord[0] = max_coord[0] - radius_act
        max_coord[1] = max_coord[1] - radius_act
        
        points.append(max_coord)
    points = np.array(points).astype(float)
    
    points[:,0] = points[:,0]*stride
    points[:,1] = points[:,1]*stride
    return points


def get_matched_pts(xml_loc, heat_loc, radius, cutoff, output_class, stride):
    all_matched_pts = []
    all_matched_preds = []
    total_nuclei = 0
    num_predicted =0

    #load heatmap
    heatmap = np.load(heat_loc)

    # get predictions and actual nuclei
    # Check if doing for variable radius:
    if type(radius)==int:
        preds = non_max_supression(heatmap, radius=radius, cutoff=cutoff, stride=stride, output_class=output_class)
    else:
        preds = non_max_supression_variable(heatmap, radius=radius, cutoff=cutoff, stride=stride, output_class=output_class)

    true_points = get_points_xml(xml_loc)

    #loop through the predictions, and check if there a corresponding true point
    # Delete the true points once they are matched, so they are not mached more than once
    tp_temp = true_points

    for index, point in enumerate(preds):
        dists = np.sqrt(np.sum((point[0:2] - tp_temp[:, 0:2]) ** 2, axis=1))
        if (len(dists)==0):
            break
        else:
            min_ind = np.argmin(dists)
            # if point has matching prediction, append it and increment the number of matched points
            if (dists[min_ind] < 10):
                all_matched_preds.append(preds[index, 2:])
                all_matched_pts.append(tp_temp[min_ind, :])
                tp_temp = np.delete(tp_temp, (min_ind), axis=0) # If the point is matched, delete from list
    acc_dict = {}
    acc_dict["all_matched_preds"] = np.array(all_matched_preds)
    acc_dict["all_matched_pts"] = np.array(all_matched_pts)
    acc_dict["total_nuclei"] = len(true_points)
    acc_dict["num_predicted"] = len(preds)
    acc_dict["abs_error"] = abs(len(true_points)-len(preds))
    return acc_dict   



def test_heat_preds(test_folder, heat_folder, radius, cutoff, output_class, stride):
    all_files=glob.glob(os.path.join(test_folder, '*'))
    all_xml = [loc for loc in all_files if 'key' in loc]
    all_matched_pts = []
    all_matched_preds = []
    abs_error_list = []
    total_nuclei = 0
    num_predicted =0

    for xml_loc in all_xml:
        image_name = xml_loc.rsplit('.', 1)[-2].rsplit('/', 1)[1].rsplit('.', 1)[0].rsplit('_', 1)[0]
        heat_loc = os.path.join(heat_folder, image_name+'_crop.npy')

        acc = get_matched_pts(xml_loc, heat_loc, radius, cutoff, output_class, stride)

        # Update 
        all_matched_preds.extend(np.array(acc["all_matched_preds"]))
        all_matched_pts.extend(np.array(acc["all_matched_pts"]))

        total_nuclei = total_nuclei + acc['total_nuclei']
        num_predicted = num_predicted + acc['num_predicted']
        abs_error_list.append(acc['abs_error'])

    acc_dict = {}
    acc_dict["all_matched_preds"] = np.array(all_matched_preds)
    acc_dict["all_matched_pts"] = np.array(all_matched_pts)
    acc_dict["total_nuclei"] = total_nuclei
    acc_dict["num_predicted"] = num_predicted
    acc_dict["abs_error"] = np.mean(abs_error_list)

    return acc_dict

# def test_heat_preds_variable(test_folder, heat_folder, radius, cutoff, output_class):
#     all_files=glob.glob(os.path.join(test_folder, '*'))
#     all_xml = [loc for loc in all_files if 'key' in loc]
#     all_matched_pts = []
#     all_preds = []
#     abs_error_list = []
#     total_nuclei = 0
#     num_predicted =0

#     for xml_loc in all_xml:
#         image_name = xml_loc.rsplit('.', 1)[-2].rsplit('/', 1)[1].rsplit('.', 1)[0].rsplit('_', 1)[0]
#         heat_loc = os.path.join(heat_folder, image_name+'_crop.npy')

#         #load heatmap
#         heatmap = np.load(heat_loc)

#         # get predictions and actual nuclei
#         preds = non_max_supression_variable(heatmap, radius=radius, cutoff=cutoff, output_class=output_class)
#         true_points = get_points_xml(xml_loc)

#         #loop through the actual points, and check if there a corresponding prediction
#         # Delete the true points once they are matched, so they are not mached more than once
#         matched=0
#         for index, point in enumerate(true_points):
#             dists = np.sqrt(np.sum((preds[:,0:2] - point[0:2]) ** 2, axis=1))
#             min_ind = np.argmin(dists)
#             # if point has matching prediction, append it and increment the number of matched points
#             if (dists[min_ind] < 10):
#                 all_preds.append(preds[min_ind, 2])
#                 all_matched_pts.append(true_points[index, :])
#                 true_points[index, 0] = -100 # If the point is matched, set the coordinate to far away, like removing from list
#                 matched=matched+1

#         # Update the counters
#         total_nuclei = total_nuclei + len(true_points)
#         num_predicted = num_predicted + len(preds)
#         abs_error_list.append(abs(len(true_points)-len(preds)))
        
#     acc_dict = {}
#     acc_dict["all_preds"] = np.array(all_preds)
#     acc_dict["all_matched_pts"] = np.array(all_matched_pts)
#     acc_dict["total_nuclei"] = total_nuclei
#     acc_dict["num_predicted"] = num_predicted
#     acc_dict["abs_error"] = np.mean(abs_error_list)
#     return acc_dict


def predict_points(point_list, model, image, im_size):
    delta=int((im_size)/2)
    image = image/255.0 # During training the images were normalized
    image = np.lib.pad(image, ((delta, delta), (delta, delta), (0,0)), 'constant', constant_values=(0, 0))
    new_preds = point_list

    for index, point in enumerate(point_list):
        if (point[0]<0 or point[1]<0):
            print("error, skipping point: ", point)
            continue
        row = int(point[0]+delta)
        col = int(point[1]+delta)
        seg_image = image[row-delta:row+delta, col-delta:col+delta, :]
        seg_image = np.expand_dims(seg_image, axis=0) # keras expects batchsize as index 0
        pred = model.predict(seg_image, batch_size=1, verbose=0)
        pred = np.argmax(pred[0][1:])+1
        new_preds[index, 2] = pred
    return new_preds


def get_matched_pts2(true_points, preds, radius, cutoff, output_class, stride, im_size):
    all_matched_pts = []
    all_matched_preds = []
    total_nuclei = 0
    num_predicted =0

    #loop through the predictions, and check if there a corresponding true point
    # Delete the true points once they are matched, so they are not mached more than once
    tp_temp = true_points
    for index, point in enumerate(preds):
        dists = np.sqrt(np.sum((point[0:2] - tp_temp[:, 0:2]) ** 2, axis=1))
        if (len(dists)==0):
            break
        else:
            min_ind = np.argmin(dists)
            # if point has matching prediction, append it and increment the number of matched points
            if (dists[min_ind] < 10):
                all_matched_preds.append(preds[index, 2:])
                all_matched_pts.append(tp_temp[min_ind, :])
                tp_temp = np.delete(tp_temp, (min_ind), axis=0) # If the point is matched, delete from list
    acc_dict = {}
    acc_dict["all_matched_preds"] = np.array(all_matched_preds)
    acc_dict["all_matched_pts"] = np.array(all_matched_pts)
    acc_dict["total_nuclei"] = len(true_points)
    acc_dict["num_predicted"] = len(preds)
    acc_dict["abs_error"] = abs(len(true_points)-len(preds))
    return acc_dict   


def test_heat_preds_2stage(test_folder, heat_folder, model, radius, cutoff, output_class, stride, im_size):
    all_files=glob.glob(os.path.join(test_folder, '*'))
    all_xml = [loc for loc in all_files if 'key' in loc]
    all_matched_pts = []
    all_matched_preds = []
    abs_error_list = []
    total_nuclei = 0
    num_predicted =0

    for xml_loc in all_xml:
        image_name = xml_loc.rsplit('.', 1)[-2].rsplit('/', 1)[1].rsplit('.', 1)[0].rsplit('_', 1)[0]
        heat_loc = os.path.join(heat_folder, image_name+'_crop.npy')
        test_loc = os.path.join(test_folder, image_name+'_crop.tif')

        heat = np.load(heat_loc)
        image = np.asarray(Image.open(test_loc))
        print('test_loc ', test_loc)
        print('image.shape', image.shape)


        # get predictions and actual nuclei
        true_points = get_points_xml(xml_loc)
        point_list = non_max_supression(heatmap=heat, radius=radius, cutoff = cutoff, stride = stride)
        preds = predict_points(point_list, model, image, im_size)
        true_pts = get_points_xml(xml_loc)

        acc = get_matched_pts2(true_points, preds, radius, cutoff, output_class, stride, im_size)
        # Update 
        all_matched_preds.extend(np.array(acc["all_matched_preds"]))
        all_matched_pts.extend(np.array(acc["all_matched_pts"]))

        total_nuclei = total_nuclei + acc['total_nuclei']
        num_predicted = num_predicted + acc['num_predicted']
        abs_error_list.append(acc['abs_error'])

    acc_dict = {}
    acc_dict["all_matched_preds"] = np.array(all_matched_preds)
    acc_dict["all_matched_pts"] = np.array(all_matched_pts)
    acc_dict["total_nuclei"] = total_nuclei
    acc_dict["num_predicted"] = num_predicted
    acc_dict["abs_error"] = np.mean(abs_error_list)

    return acc_dict




