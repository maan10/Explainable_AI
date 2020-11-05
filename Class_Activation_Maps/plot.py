# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 15:32:30 2020

@author: Vanditha Rao

This script allows the user to plot the input image with top k prediction 
predicted by the model. Image file can be of any format.

This script requires that tensorflow be installed within the python environment
you are running this script in.

Here, tensorflow version 2.2, and python version
3.7.7 is used.

This file is imported as a module and contains the following functions:

    * plot_image 
        
        This function is used to plot the input image with the true label
        and the prediction of the true label
        
    * plot_value_array
    
        This function is used to plot the horizontal bar plot of top k
        predicitons from the model
        
        
"""

import tensorflow 

def plot_image(predictions_array,
               true_label,
               original_img,
               classes,
               ax):
    
    """
    Plots the input image with the true label and the prediction of the
    true label 
        
    Args:
    
    original_img: str
    The input image filepath
    
    predictions_array: numpy.ndarray
    numpy array of output predictions for the input image
    
    true_label: int
    the true label of the input image
    
    classes: list
    classes in the given dataset
    
    ax:
    plot the image on the axes ax     
       
    """
     
    ax.set_xticks([])
    ax.set_yticks([])
    
    ax.imshow(original_img, aspect = 'equal')

    ax.set_xlabel("True label: {} ({:2.0f}%)".format(classes[true_label],
                                                     100*predictions_array[true_label]),
                  
                  color='green', va = 'top', ha = 'center', labelpad = 2, fontsize = 10, fontname = 'Times New Roman')
     
def plot_value_array(predictions_array,
                     true_label,
                     classes,
                     ax,
                     k):
    
    """
    Plot the horizontal bar plot of top k predicitons from the model
    
    Args:
    
    predictions_array: numpy.ndarray
    numpy array of output predictions for the input image
    
    true_label: int
    the true label of the input image
    
    classes: list
    classes in the given dataset
    
    ax:
    plot the image on the axes ax   
    
    k: int
    Number of top elements to look for 
    
    """
    
    prediciton_k_values, top_k_indices = tensorflow.nn.top_k(predictions_array, k)
    prediciton_k_values, top_k_indices = prediciton_k_values.numpy(), top_k_indices.numpy()

    class_top_3 = []

    for i in range(k):
        class_top_3.append(classes[top_k_indices[i]])

    ax.tick_params(axis= 'y',
                    labelsize=10,
                    length = 0,
                    which = 'both',
                    labelright= True,
                    pad=-1,
                    direction="in",
                    right= True,
                  )
    
    ax.patch.set_facecolor('xkcd:light blue')
    ax.set_xticks([])
    ax.set_yticks(range(k))
    ax.set_yticklabels(class_top_3, horizontalalignment='right', verticalalignment='center', fontname = 'Times New Roman')
    ax.yaxis.tick_right()

    width = 0.75 
    thisplot = ax.barh(range(k), prediciton_k_values, width, align='center')

    for i, v in enumerate(prediciton_k_values):
        ax.text(0.005, i, '{:2.0f}%' .format(v*100),
                color='black',
                fontweight='bold',
                )
        
    for i in range(len(top_k_indices)):
        thisplot[i].set_color('red')
        if top_k_indices[i]==true_label:
            thisplot[i].set_color('green')
            
    ax.invert_yaxis() 