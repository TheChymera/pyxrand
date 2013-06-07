#!/usr/bin/env python
from __future__ import division
__author__ = 'Horea Christian'
from os import path, listdir, walk
from scipy import ndimage
from scipy.misc import imsave
import numpy as np
import matplotlib.cm as cm
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

max_randomness = 16
randomness_steps = 8

localpath = '~/src/picrand/img/'
localpath = path.expanduser(localpath)

input_folder = localpath
randomness_step = int(max_randomness / randomness_steps)

def shift_funct(output_coords,rdness):
    return (output_coords[0] + np.random.randint(-rdness*randomness_step, rdness*randomness_step, (1, 1)), output_coords[1] + np.random.randint(-rdness*randomness_step, rdness*randomness_step, (1, 1)))
    
for pic in listdir(input_folder):
    if path.splitext(pic)[0][-2:] == 'rd':
	pass
    else:
	im = mpimg.imread(input_folder+pic)
	for rdness in np.arange(randomness_steps)+1:
	    im = ndimage.geometric_transform(im, shift_funct, mode= 'nearest', extra_arguments=(rdness,))	
	    imsave(input_folder+path.splitext(pic)[0]+'_'+str(rdness*randomness_step)+'rd.jpg', im)
    
