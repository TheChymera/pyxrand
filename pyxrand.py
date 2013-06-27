#!/usr/bin/env python
from __future__ import division
__author__ = 'Horea Christian'
from os import path, listdir
from scipy import ndimage
from scipy.misc import imsave
from skimage.util.shape import view_as_windows
#~ PYTHONPATH=sys.path.insert(0, 'home/chymera/src/scikit-image/skimage/util/') montage.py montage2d
import numpy as np
import matplotlib.cm as cm
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import imp
montage = imp.load_source('montage2d', '/home/chymera/src/scikit-image/skimage/util/montage.py')

by_pixel = False

cell_size_step = 16
cell_size_minimum = 16
cell_size_increments = 8

max_randomness = 16 # type maximal re-mapping radius -- ONLY RELEVANT FOR by_pixel == True
randomness_steps = 8 # type desired number of randomness steps (i.e. the number of output files) -- ONLY RELEVANT FOR by_pixel == True
localpath = '~/src/pyxrand/img/' # path where image files are located

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
	    imsave(input_folder+path.splitext(pic)[0]+'_'+str(rdness*randomness_step)+'px-rand.jpg', im)
    else:
	for cell_increment in np.arange(cell_size_increments):
	    cell_size = cell_size_minimum+cell_size_step*cell_increment
	    im = mpimg.imread(input_folder+pic)
	    height, width = np.shape(im)
	    record_squares_switch = True
	    record_rowstart_switch = True
	    counter = 0
	    slice_coordinates = np.zeros(2)
	    slices = np.zeros((cell_size, cell_size))
	    
	    # calculate subimage to make sure that pixels exceeding the optimal slice are distributed equally along x and y  	    
	    nonzero_y = np.shape([line for line in im if len(np.unique(line)) >= 10])[0] # gets the number of lines with more than background values
	    nonzero_x = np.shape([line for line in im.T if len(np.unique(line)) >= 10])[0] # gets the number of columns with more than background values
	    leadingzeros_y = 0
	    leadingzeros_x = 0
	    for y in im:
		if len(np.unique(y)) < 10:
		    leadingzeros_y +=1
		else: 
		    break
	    
	    for x in im.T:
		if len(np.unique(x)) < 10:
		    leadingzeros_x +=1
		else: 
		    break
	    rest_y_r = np.floor((cell_size-(nonzero_y % cell_size)) / 2)
	    rest_y_l = np.ceil((cell_size-(nonzero_y % cell_size)) / 2)
	    rest_x_r = np.floor((cell_size-(nonzero_x % cell_size)) / 2)
	    rest_x_l = np.ceil((cell_size-(nonzero_x % cell_size)) / 2)
	    sub_im = im[leadingzeros_y-rest_y_l:leadingzeros_y+nonzero_y+rest_y_r,leadingzeros_x-rest_x_l:leadingzeros_x+nonzero_x+rest_x_r]
	    # end subimage
	    
	    squares = view_as_windows(sub_im, (cell_size, cell_size))
	    cell_squares = squares[::cell_size,::cell_size]
	    print np.shape(im), np.shape(sub_im), np.shape(squares), np.shape(cell_squares), np.shape(cell_squares)[:2]
	    cell_squares_random = np.random.permutation(np.reshape(cell_squares, (-1,cell_size,cell_size)))
	    leimage = montage.montage2d(cell_squares_random, output_shape=(np.shape(cell_squares)[:2]))
	    
	    pad_value = np.mean(im[:cell_size,:cell_size]).astype(int)
	    print pad_value
	    leimage = np.pad(leimage, ((leadingzeros_y-rest_y_l,np.shape(im)[0]-(leadingzeros_y+nonzero_y+rest_y_r)),(leadingzeros_x-rest_x_l,np.shape(im)[1]-(leadingzeros_x+nonzero_x+rest_x_r))), 'constant' ,constant_values=pad_value)
	    
	    print np.shape(im)
	    
	    #~ imgplot = plt.imshow(leimage, cmap = cm.Greys_r, interpolation='nearest')
	    imsave(input_folder+path.splitext(pic)[0]+'_'+str(cell_size)+'cell-rand.jpg', leimage)
    
