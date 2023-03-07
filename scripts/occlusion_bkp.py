#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This scripts occludes images

# Steps:
# imports objects, and runs 3 manipulations
# on two levels of difficulty (easy/hard)
# and 2 ways of occluding (many small occluders/
# few large occluders)

import cv2 as cv
import numpy as np
import pathlib
import glob
import os
from numpy.random import randint
from tqdm import tqdm

def occlude(
    img_dir,
    easy = 20,
    hard = 60,
    control=False, 
    blobs = True, 
    deletion = True,
    partialviewing = True, 
    many_small = True,
    few_large = True,
    col = 0
    ):
    """
    Apply occlusion to a set of images by adding blobs or occluding part of the image.

    Args:
        img_dir (str): The path to the directory containing the images to occlude.
        easy (int, optional): The percentage of the object *occluded* in the low level of occlusion. 
            Defaults to 20.
        hard (int, optional): The percentage of the object *occluded* in the high level of occlusion. 
            Defaults to 60.
        many_small (bool, optional): If True, apply many small occluders, otherwise apply few large occluders. 
            Defaults to True.
        few_large (bool, optional): If True, apply few large occluders, otherwise apply many small occluders. 
            Defaults to True.
        col (int, optional): The grayscale color of the occluding window. Defaults to 0 (black).

    Returns:
        None: Saves the occluded images in the output directory.
    """
    np.random.seed()

    imagePath = glob.glob(img_dir + '/*') # take in all the files in there
    imagePath.sort() # make sure the files are sorted

    # list the output paths
    listOutputPaths = []
    
    # name the output paths
    if deletion:
        del_outdir = r'./output/deletion/'
        listOutputPaths.append(del_outdir)
    if blobs:
        blobs_outdir = r'./output/blobs/'
        listOutputPaths.append(blobs_outdir)
    if partialviewing:
        pv_outdir = r'./output/partialviewing/'
        listOutputPaths.append(pv_outdir)

    # create the necessary output folders
    for manipulation in listOutputPaths:
        if not os.path.exists(manipulation):
            os.makedirs(manipulation)

    # decide on the portion of object **occluded** in high and low **levels of occlusion**
    occlusion_low = easy
    occlusion_high = hard
    
    # if manySmall:
    # enter parameters for many small
    ms_sizeOccluder = [10, 40]
    ms_nOccluders_low = 15
    ms_nOccluders_high = 50
        
    # enter parameters for few large
    # elif fewLarge:
    fl_sizeOccluder = [70, 300]
    fl_nOccluders_low = 5
    fl_nOccluders_high = 5
    
    # decide on the colors to use, black by default
    colBlob = (col, col, col) # color of the occluding blobs
    
    # find the BGR color of the background from the upper left-most pixel
    colBckgrnd = ((cv.imread(imagePath[0], -1))[0,0,0:3])
    colBckgrnd = int(colBckgrnd[0]), int(colBckgrnd[1]), int(colBckgrnd[2])
    print('Color background is', colBckgrnd)
    
    # deletion

    if deletion:
        for size_occl in tqdm({'manysmall', 'fewlarge'}):
            
            writePath = del_outdir + size_occl
            
            if (size_occl == 'manysmall') & (many_small):
                sizeOccluder = ms_sizeOccluder
                nOccluders_low = ms_nOccluders_low
                nOccluders_high = ms_nOccluders_high
                
            elif (size_occl == 'fewlarge') & (few_large):
                sizeOccluder = fl_sizeOccluder
                nOccluders_low = fl_nOccluders_low
                nOccluders_high = fl_nOccluders_high

            for i in range(len(imagePath)):
                
                # read the file
                file = pathlib.Path(imagePath[i])
                # read the image
                im = cv.imread(str(file), -1)

                # creating object masks & measuring pixel size
                # creating a white mask where the object is located
                object_mask = np.zeros((im.shape[0], im.shape[1]), np.uint8)
                object_mask[im[:,:,3]==255] = 255
                # measuring the pixel size of the object
                size_object = object_mask[object_mask==255].shape[0]

                # removing transparency in the background and make image 2D
                im[:,:,3] == 255

                # operate the occlusion for each level
                for level in ('low', 'high'):
                    
                    # create the write path, one for each level
                    out_dir = writePath + "/{}/".format(level)
                    if not os.path.exists(out_dir): os.makedirs(out_dir)

                    if level == 'low':
                        nOccluders = nOccluders_low
                        occlusion_level = occlusion_low
                    elif level == 'high':
                        nOccluders = nOccluders_high
                        occlusion_level = occlusion_high

                    proportion_occluded = 0

                    # as long as the difference between the required occlusion proportion and the actual occlusion proportion is more than 1%    
                    while abs(occlusion_level - proportion_occluded) > 1:
                        
                        # make a 2D copy of the image
                        gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
                        
                        # create the coordinates for occluders
                        points_1 = np.array(randint(0, gray.shape[0], nOccluders))
                        points_2 = np.array(randint(0, gray.shape[1], nOccluders))
                        points = np.column_stack((points_1,points_2))

                        # randomly generate the blob radiuses within the determined range
                        radiuses = randint(sizeOccluder[0], sizeOccluder[1], nOccluders)

                        # preparing the background for occluder mask
                        # create the black figure that will hold the occluder mask
                        occluder_mask = np.zeros((gray.shape[0], gray.shape[1]), np.uint8)

                        # drawing occluders on the image & on their mask
                        for point, radius in zip(points, radiuses):
                            cv.circle(gray, point, radius, tuple(colBckgrnd[0:3]), -1)
                            # here I draw the occluder mask
                            cv.circle(occluder_mask, point, radius, color = (255, 255, 255), thickness = -1)

                        # calculate the intersection between masks
                        intersection = cv.bitwise_and(object_mask, occluder_mask) # create the intersection
                        intersectionSize = intersection[intersection==255].shape[0] # measure the size of intersection
                        proportion_occluded = ((intersectionSize/size_object)*100)

                        # if the actual occlusion is less than 1% away from goal, write the obtained image
                        if abs(occlusion_level - proportion_occluded) < 1:
                            cv.imwrite(out_dir + '{}_deletion_{}'.format(size_occl, level) + file.name, gray)
                            # print('Object occluded successfully')
                            break
                        # if occlusion is too low, increase the number of occluders
                        elif (occlusion_level > proportion_occluded):
                            nOccluders += 1
                            # print('Occlusion too low: {} instead of {}.'.format(proportion_occluded, occlusion_level))
                            continue
                        # if occlusion is too high, decrease the number of occluders
                        elif (occlusion_level < proportion_occluded):
                            nOccluders -= 1
                            # print('Occlusion too high: {} instead of {}.'.format(proportion_occluded, occlusion_level))
                            continue


    # partial viewing

    if partialviewing:
            for size_occl in tqdm({'manysmall', 'fewlarge'}):
                
                writePath = pv_outdir + size_occl
                
                if (size_occl == 'manysmall') & (many_small):
                    sizeOccluder = ms_sizeOccluder
                    nOccluders_low = ms_nOccluders_low
                    nOccluders_high = ms_nOccluders_high
                    
                elif (size_occl == 'fewlarge') & (few_large):
                    sizeOccluder = fl_sizeOccluder
                    nOccluders_low = fl_nOccluders_low
                    nOccluders_high = fl_nOccluders_high

                # for i in tqdm(range(len(imagePath))):
                for i in range(len(imagePath)):

                    # get the file
                    file = pathlib.Path(imagePath[i])
                    # read the image
                    im = cv.imread(str(file), -1)

                    # creating a mask where the object is located
                    object_mask = np.zeros((im.shape[0], im.shape[1]), np.uint8)
                    object_mask[im[:,:,3]==255] = 255
                    # measuring the pixel size of the object
                    size_object = object_mask[object_mask==255].shape[0]

                    # removing transparency in the background and make image 2D
                    im[:,:,3] == 255

                    # define the levels based on control or not
                    if control:
                        levels = ('low', 'high', 'control')
                    else:
                        levels = ('low', 'high')
                    
                    # operate the occlusion for each level
                    for level in levels:

                        # create the write path, one for each level
                        out_dir = writePath + "/{}/".format(level)
                        if not os.path.exists(out_dir): os.makedirs(out_dir)
                        # give the number of occluders and proportion of occlusion based on the level
                        if level == 'high':
                            nOccluders = nOccluders_low
                            occlusion_level = occlusion_low
                        elif (level == 'low') | (level == 'control'):
                            nOccluders = nOccluders_high
                            occlusion_level = occlusion_high

                        proportion_occluded = 0

                        while abs(occlusion_level - proportion_occluded) > 1:

                            # make a 2D copy of the image
                            gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

                            # create the coordinates for occluders
                            points_1 = np.array(randint(0, gray.shape[0], nOccluders))
                            points_2 = np.array(randint(0, gray.shape[1], nOccluders))
                            points = np.column_stack((points_1,points_2))

                            # randomly generate the blob radiuses within the determined range
                            radiuses = randint(sizeOccluder[0], sizeOccluder[1], nOccluders)

                            # create the initial full mask and set its color
                            occluder = np.zeros_like(gray)

                            # drawing occluders on the image & on their mask
                            for point, radius in zip(points, radiuses):
                                cv.circle(occluder, point, radius, (255,255,255), -1)
                            occluded = cv.bitwise_and(gray, gray, mask = occluder)
                            # turn the occluder into the desired color
                            occluded[occluder == 0] = col
                            # np.unique(occluder)
                            # from matplotlib import pyplot as plt
                            # plt.imshow(occluded, cmap = 'gray')
                            # plt.show()

                            # create the occluder mask
                            # occluder_inv = cv.bitwise_not(occluder)
                            blank = np.ones_like(gray)
                            blank[:] = 255
                            occluder_mask = cv.bitwise_and(blank, blank, mask = occluder)

                            # calculate the intersection between masks
                            intersection = cv.bitwise_and(object_mask, occluder_mask) # create the intersection
                            intersectionSize = intersection[intersection==255].shape[0] # measure the size of intersection
                            proportion_occluded = ((intersectionSize/size_object)*100)

                            # write the image only if occlusion is as required
                            if abs(occlusion_level - proportion_occluded) < 1:
                                cv.imwrite(out_dir + '{}_partialViewing_{}'.format(size_occl, level) + file.name, occluded)
                                # print('Object occluded successfully')
                                if level == 'control':
                                    mask = (cv.imread(str(file), -1))[:,:, 3] == 255
                                    occluded[mask] = im[mask][:, 2]
                                    cv.imwrite(out_dir + '{}_partialViewing_{}'.format(size_occl, level) + file.name, occluded)
                                break
                            elif (occlusion_level > proportion_occluded):
                                nOccluders += 1
                                # print('Occlusion too low: {} instead of {}.'.format(proportion_occluded, occlusion_level))
                                continue
                            elif (occlusion_level < proportion_occluded):
                                nOccluders -= 1
                                # print('Occlusion too high: {} instead of {}.'.format(proportion_occluded, occlusion_level))
                                continue

    # blobs

    if blobs:
        for size_occl in tqdm({'manysmall', 'fewlarge'}):
            
            writePath = blobs_outdir + size_occl
                    
            if (size_occl == 'manysmall') & (many_small):
                sizeOccluder = ms_sizeOccluder
                nOccluders_low = ms_nOccluders_low
                nOccluders_high = ms_nOccluders_high
                
            elif (size_occl == 'fewlarge') & (few_large):
                sizeOccluder = fl_sizeOccluder
                nOccluders_low = fl_nOccluders_low
                nOccluders_high = fl_nOccluders_high

            for i in range(len(imagePath)):
                
                # read the file
                file = pathlib.Path(imagePath[i])
                # read the image
                im = cv.imread(str(file), -1)

                # creating object masks & measuring pixel size
                # creating a white mask where the object is located
                object_mask = np.zeros((im.shape[0], im.shape[1]), np.uint8)
                object_mask[im[:,:,3]==255] = 255
                # measuring the pixel size of the object
                size_object = object_mask[object_mask==255].shape[0]

                # removing transparency in the background and make image 2D
                im[:,:,3] == 255

                # define the levels based on control or not
                if control:
                    levels = ('low', 'high', 'control')
                else:
                    levels = ('low', 'high')
                    
                # operate the occlusion for each level
                for level in levels:
                    
                    # create the write path, one for each level
                    out_dir = writePath + "/{}/".format(level)
                    if not os.path.exists(out_dir): os.makedirs(out_dir)

                    if (level == 'low') | (level == 'control'):
                        nOccluders = nOccluders_low
                        occlusion_level = occlusion_low
                    elif level == 'high':
                        nOccluders = nOccluders_high
                        occlusion_level = occlusion_high

                    proportion_occluded = 0

                    # as long as the difference between the required occlusion proportion and the actual occlusion proportion is more than 1%    
                    while abs(occlusion_level - proportion_occluded) > 1:
                        
                        # make a 2D copy of the image
                        gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
                        
                        # create the coordinates for occluders
                        points_1 = np.array(randint(0, gray.shape[0], nOccluders))
                        points_2 = np.array(randint(0, gray.shape[1], nOccluders))
                        points = np.column_stack((points_1,points_2))

                        # randomly generate the blob radiuses within the determined range
                        radiuses = randint(sizeOccluder[0], sizeOccluder[1], nOccluders)

                        # preparing the background for occluder mask
                        # create the black figure that will hold the occluder mask
                        occluder_mask = np.zeros((gray.shape[0], gray.shape[1]), np.uint8)

                        # drawing occluders on the image & on their mask
                        for point, radius in zip(points, radiuses):
                            cv.circle(gray, point, radius, tuple(colBlob[0:3]), -1)
                            # here I draw the occluder mask
                            cv.circle(occluder_mask, point, radius, color = (255, 255, 255), thickness = -1)

                        # calculate the intersection between masks
                        intersection = cv.bitwise_and(object_mask, occluder_mask) # create the intersection
                        intersectionSize = intersection[intersection==255].shape[0] # measure the size of intersection
                        proportion_occluded = ((intersectionSize/size_object)*100)

                        # if the actual occlusion is less than 1% away from goal, write the obtained image
                        if abs(occlusion_level - proportion_occluded) < 1:
                            cv.imwrite(out_dir + '{}_blobs_{}'.format(size_occl, level) + file.name, gray)
                            if control:
                            # print('Object occluded successfully')
                                mask = (cv.imread(str(file), -1))[:,:, 3] == 255
                                gray[mask] = im[mask][:, 2]
                                cv.imwrite(out_dir + '{}_blobs_{}'.format(size_occl, level) + file.name, gray)
                            break
                        # if occlusion is too low, increase the number of occluders
                        elif (occlusion_level > proportion_occluded):
                            nOccluders += 1
                            # print('Occlusion too low: {} instead of {}.'.format(proportion_occluded, occlusion_level))
                            continue
                        # if occlusion is too high, decrease the number of occluders
                        elif (occlusion_level < proportion_occluded):
                            nOccluders -= 1
                            # print('Occlusion too high: {} instead of {}.'.format(proportion_occluded, occlusion_level))
                            continue
