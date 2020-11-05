# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 20:41:50 2020

@author: Vanditha Rao

This script allows the user to plot the input image with top k prediction
predicted by the model, augmented image and the image with heatmap.
Image file can be of any format.

This script requires that matplotlib be installed within the python environment
you are running this script in.

Here, matplotlib version 3.1.3 and python version 3.7.7 is used.

This file is imported as a module and contains the following functions:
        
    * heatmap_display
    
        This function create three subplots.
        Subplot 1 calls plot_image function, subplot 2 calls plot_value_array
        function and subplot 3 displays the output image. 
        
        To create more flexibility in creating an axes object at a specific
        location of the grid, the script uses subplot2grid() function from
        Matplotlib

"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from plot import plot_image, plot_value_array

def heatmap_display(predictions_array,
                    true_label,
                    original_img,
                    aug_img,
                    heatmap,
                    classes,
                    k = None,
                    colormap = 'viridis',
                    save_name = None):
    
  
    """
    Args:
    
    original_img: str
    The input image filepath
    
    aug_img: str
    The augmented image filepath
    
    predictions_array: numpy.ndarray
    numpy array of output predictions for the input image
    
    true_label: int
    the true label of the input image
    
    heatmap: numpy.ndarray
    sensitivity maps with shape of dimension of the img
    
    classes: list
    classes in the given dataset
    
    k: int (default = None)
    Number of top elements to look for
    
    colormap: str (default= 'viridis')   
    The Colormap instance or registered colormap name used to map scalar data
    to colors. Colormaps is chosen from matplotlib
    
    save_name: str (default = None)
    the output filename
    
    """
    
    if k == None:
        k = len(classes)
    
    fig = plt.figure(figsize=(2,2), constrained_layout=False)

    # ------------------------------------------------------------------------

    ax1 = plt.subplot2grid((4, 6), (0, 0), rowspan=3, colspan=2, frameon=False)
    ax1.yaxis.set_major_locator((plt.NullLocator()))
    ax1.xaxis.set_major_locator((plt.NullLocator()))
    
    plot_image(predictions_array, true_label, original_img, classes, ax = ax1)
    

    # ------------------------------------------------------------------------

    ax2 = plt.subplot2grid((4, 6), (3, 0), rowspan=1, colspan=2)
    ax2.yaxis.set_major_locator((plt.NullLocator()))
    ax2.xaxis.set_major_locator((plt.NullLocator()))
    
    plot_value_array(predictions_array, true_label, classes, ax = ax2, k = k)
    
    ax2.set_xlabel('(a)', fontsize = 14, fontname = 'Times New Roman')

    # ------------------------------------------------------------------------
    
    ax3 = plt.subplot2grid((4, 6), (0, 2), rowspan=4, colspan=2, frameon=False,
                           xticklabels=[], yticklabels=[], xticks=[], yticks=[]
                          )
    
    ax3.yaxis.set_major_locator((plt.NullLocator()))
    ax3.xaxis.set_major_locator((plt.NullLocator()))
    ax3.imshow(aug_img, aspect = 'equal')
    ax3.set_xlabel('(b)', fontsize = 14, fontname = 'Times New Roman')

    # ------------------------------------------------------------------------
    
    ax4 = plt.subplot2grid((4, 6), (0, 4), rowspan=4, colspan=2,
                           xticklabels=[], yticklabels=[], xticks=[],
                           yticks=[],
                           frameon=False)
    
    ax4.yaxis.set_major_locator((plt.NullLocator()))
    ax4.xaxis.set_major_locator((plt.NullLocator()))
    ax4.imshow(np.squeeze(original_img), alpha = 1)
    
    hm = ax4.imshow(heatmap, alpha = 0.7, cmap=colormap)

    divider = make_axes_locatable(ax4)
    cax2 = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(hm, cax=cax2) 
                 
    ax4.set_xlabel('(c)', labelpad = 36, fontsize = 14,
                   fontname = 'Times New Roman')
    
    fig.set_size_inches(w=15, h=5)
    
    if save_name != None:
        fig.savefig(save_name,
                    bbox_inches = 'tight', pad_inches = 0)
    
    plt.show()
