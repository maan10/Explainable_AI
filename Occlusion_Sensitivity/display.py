# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 20:41:50 2020

@author: Vanditha Rao
"""

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import tensorflow 
# from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker

def plot_image(predictions_array,
               true_label,
               original_img,
               classes,
               ax):
     
    ax.set_xticks([])
    ax.set_yticks([])
    
    ax.imshow(original_img, aspect = 'equal')

    ax.set_xlabel("True label: {} ({:2.0f}%)".format(classes[true_label],
                                                     100*predictions_array[true_label]),
                  
                  color='green', va = 'top', ha = 'center', labelpad = 2, fontsize = 10, fontname = 'Times New Roman')
    
    ### To display the predicted label and the true label in two different
    ### colors below the image (comment ax.set_xlabel)
    
    # predicted_label = np.argmax(predictions_array)
    #if predicted_label == true_label:
    #    color = 'g'
    #else:
    #    color = 'r'
        
    #xbox1 = TextArea("{} {:2.0f}%" .format(classes[predicted_label], 100*np.max(predictions_array)),
    #                 textprops=dict(color=color, size=12, ha='left',va='top'))
    
    #xbox2 = TextArea("({} {:2.0f}%)" .format(classes[true_label], 100*predictions_array[true_label]),
    #                 textprops=dict(color="g", size=12, ha='left',va='top'))
    
    #xbox = HPacker(children=[xbox1, xbox2],align="bottom", pad=0., sep=2)

    #anchored_xbox = AnchoredOffsetbox(loc=8, child=xbox, pad=-0.5, frameon=False, # bbox_to_anchor=(0.08, 0.4), 
    #                              bbox_transform=ax.transAxes, borderpad=0.)

    #ax.add_artist(anchored_xbox)
    

def plot_value_array(predictions_array,
                     true_label,
                     classes,
                     ax,
                     k):
    
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
    
def heatmap_display(predictions_array,
                    true_label,
                    original_img,
                    aug_img,
                    sensitivity_map,
                    classes,
                    colormap = 'viridis',
                    k = None,
                    dpi = 80,
                    save_name = None):
    
    if k == None:
        k = len(classes)
    
    fig = plt.figure(figsize=(2,2), dpi=dpi, constrained_layout=False)

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
    hm = ax4.imshow(sensitivity_map, alpha = 0.7, cmap=colormap)

    divider = make_axes_locatable(ax4)
    cax2 = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(hm, cax=cax2) 
                 
    ax4.set_xlabel('(c)', labelpad = 36, fontsize = 14, fontname = 'Times New Roman')
    
    fig.set_size_inches(w=15, h=5)
    
    if save_name != None:
        fig.savefig(save_name, dpi=dpi,
                    bbox_inches = 'tight', pad_inches = 0)
    
    plt.show()