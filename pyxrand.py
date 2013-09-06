#!/usr/bin/env python
from __future__ import division
__author__ = 'Horea Christian'
from os import path, listdir
from scipy import ndimage
from scipy.misc import toimage
from skimage.util.shape import view_as_windows
from skimage.util.montage import montage2d
import numpy as np
import matplotlib.cm as cm
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2

by_pixel = False # True if you want to shuffle by-pixel, False if you want to shuffle by cluster.
localpath = '~/src/pyxrand/img/' # path where image files are located

cell_size_step = 4 # in what steps should the cell size increase [px] ?
cell_size_minimum = 6 # what's the minimum cell size / start cell size [px] ?
cell_size_increments = 1 # how many pictures do you want ?

max_randomness = 16 # type maximal re-mapping radius -- ONLY RELEVANT FOR by_pixel == True
randomness_steps = 8 # type desired number of randomness steps (i.e. the number of output files) -- ONLY RELEVANT FOR by_pixel == True

column_tolerance = 6 # the columns are the first step in ROI selection. This setting to accounts for slightly fuzzy background
row_tolerance = 2 # the columns are the second step in ROI selection. This setting to accounts for slightly fuzzy background, is extra small because for small clusters equal-color lines may occur in the face region

localpath = path.expanduser(localpath)
input_folder = localpath

for pic in listdir(input_folder):
    if path.splitext(pic)[0][-4:] == 'rand': # identifies output images 
	continue				   # don't re-randomize them!
    elif by_pixel:
	randomness_step = int(max_randomness / randomness_steps)
	def randomization_funct(output_coords,rdness): 
	    return (output_coords[0] + np.random.randint(-rdness*randomness_step, rdness*randomness_step, (1, 1)), output_coords[1] + np.random.randint(-rdness*randomness_step, rdness*randomness_step, (1, 1)))
	im = mpimg.imread(input_folder+pic)
	for rdness in np.arange(randomness_steps)+1:
	    im = ndimage.geometric_transform(im, randomization_funct, mode= 'nearest', extra_arguments=(rdness,))
	    toimage(im, cmin=0, cmax=255).save(input_folder+path.splitext(pic)[0]+'_px'+str(rdness*randomness_step)+'rand.jpg') # use this instead of imsave to avoide rescaling to maximal dynamic range
    else:
	print(pic)
	for cell_increment in np.arange(cell_size_increments):
	    cell_size = cell_size_minimum+cell_size_step*cell_increment
	    im = mpimg.imread(input_folder+pic)
	    try:
		height, width = np.shape(im)
	    except ValueError:
		print('The script currently only supports grayscale images - this is probably RGB. It will not be processed to clusters.')
		continue
		
	    slice_coordinates = np.zeros(2)
	    slices = np.zeros((cell_size, cell_size))
	    
	    # calculate subimage to make sure that pixels exceeding the optimal slice are distributed equally along x and y  	    
	    nonzero_y = np.shape([line for line in im if len(np.unique(line)) >= column_tolerance])[0] # counts the number of lines with more than background values
	    leadingzeros_y = 0
	    for y in im:
		if len(np.unique(y)) < column_tolerance: # counts the number of lines with less than background values
		    leadingzeros_y +=1
		else: 
		    break

	    rest_y_d = np.floor((cell_size-(nonzero_y % cell_size)) / 2) # pixels surplus after cluster placement within ROI (d for down)
	    rest_y_u = np.ceil((cell_size-(nonzero_y % cell_size)) / 2)
	    sub_im = im[leadingzeros_y-rest_y_u:leadingzeros_y+nonzero_y+rest_y_d,:]
	    if leadingzeros_y-rest_y_u <=0:
		print('This picture has a bad background (above the ROI). It will not be processed to clusters.')
		continue
	    # end subimage
	    
	    sub_im_rows = np.reshape(sub_im, (-1, cell_size, np.shape(sub_im)[1]))
	    
	    row_start_stop_cells = np.zeros((np.shape(sub_im_rows)[0],3)) # variable containing the x position where the first cell starts and the last cell ends
	    all_squares = np.zeros((1,cell_size,cell_size)) # first zeroes frame (created just for the vstack to work at the first iteration)
	    
	    for row_number, sub_im_row in enumerate(sub_im_rows):
		#~ print np.shape(sub_im_row)
		nonzero_x_row = np.shape([line for line in sub_im_row.T if len(np.unique(line)) >= row_tolerance])[0] # counts the number of lines with more than background values
		leadingzeros_x_row = 0
		for x in sub_im_row.T:
		    if len(np.unique(x)) < row_tolerance: # counts the number of lines with less than background values
			leadingzeros_x_row +=1
		    else: 
			break
		
		rest_x_r = np.floor((cell_size-(nonzero_x_row % cell_size)) / 2) # pixels surplus after cluster placement within ROI
		rest_x_l = np.ceil((cell_size-(nonzero_x_row % cell_size)) / 2)
			
		if leadingzeros_x_row-rest_x_l <=0:
		    print('This picture has a bad background (left of the ROI). It will not be processed to clusters.')
		    break_parentloop = True
		    break
		else:
		    sub_row = sub_im_row[:,leadingzeros_x_row-rest_x_l:leadingzeros_x_row+nonzero_x_row+rest_x_r]
		    squares = view_as_windows(sub_row, (cell_size, cell_size))
		    cell_squares = squares[:,::cell_size][0]
		    all_squares = np.vstack((all_squares, cell_squares))
		    row_start_stop_cells[row_number, 0] = leadingzeros_x_row-rest_x_l # start pos
		    row_start_stop_cells[row_number, 1] = np.shape(im)[1]-(leadingzeros_x_row+nonzero_x_row+rest_x_r) # stop pos
		    row_start_stop_cells[row_number, 2] = np.shape(cell_squares)[0] # cells number
		    break_parentloop = False
	    
	    if break_parentloop:
		continue
	    
	    all_squares = all_squares[1:] # remove first zeroes frame (created just for the vstack to work at the first iteration)
	    all_squares = np.random.permutation(all_squares)
	    	    
	    pad_value = np.mean(im[:cell_size,:cell_size]).astype(int)
	    reconstructed_im = np.ones((leadingzeros_y-rest_y_u,np.shape(im)[1])) * pad_value
	    scrambled_image = np.zeros((1, np.shape(im)[1]))
	    
	    for row_number, sub_im_row in enumerate(sub_im_rows):
		shuffled_squares = montage2d(all_squares[:row_start_stop_cells[row_number, 2]], grid_shape=(1,row_start_stop_cells[row_number, 2]))
		all_squares = all_squares[row_start_stop_cells[row_number, 2]:]
		padded_row = np.pad(shuffled_squares, ((0,0),(row_start_stop_cells[row_number,0],row_start_stop_cells[row_number,1])), 'constant' ,constant_values=pad_value)
		scrambled_image = np.vstack((scrambled_image, padded_row))
		
	    scrambled_image = scrambled_image[1:]
	    scrambled_image = np.pad(scrambled_image, ((leadingzeros_y-rest_y_u, np.shape(im)[0]-(leadingzeros_y+nonzero_y+rest_y_d)),(0,0)), 'constant' ,constant_values=pad_value)
		    
	    #~ imgplot = plt.imshow(scrambled_image, cmap = cm.Greys_r, interpolation='nearest')
	    #~ plt.show()
	    
	    toimage(scrambled_image, cmin=0, cmax=255).save(input_folder+path.splitext(pic)[0]+'_cell'+str(cell_size)+'rand.jpg') # use this instead of imsave to avoide rescaling to maximal dynamic range
	    print('Done!')
