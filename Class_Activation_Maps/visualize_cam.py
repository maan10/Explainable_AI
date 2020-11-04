# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 03:14:52 2020

@author: Vanditha Rao

This script allows the user to implement class activation maps. Image file can be of any format.
This script requires that tensorflow and OpenCV be installed within the python environment you are
running this script in.

Here, tensorflow version 2.2, cv2 version 4.2.0 and python version 3.7.7 is used.

This file is imported as a module and contains visualize_cam functions which implement class activation maps

"""

from tensorflow.keras import backend as K
import numpy as np
import cv2

def visualize_cam(model,
                  layer_name,
                  original_image,
                  colormap = cv2.COLORMAP_JET):
    
   """ 
   original_img: str
   The input image filepath
   
   
   model:
   tensorflow model
  
   layer_name: str
   The layer name of the last convolutional layer
   
   colormap: str (default= cv2.COLORMAP_JET)   
   The Colormap instance or registered colormap name used to map scalar data to colors. Colormaps is chosen from OpenCV
   
   """
    
    _, width, height, c = original_image.shape
    
    img = cv2.resize(original_image, (224,224))
    aug_img = next(datagen.flow(np.expand_dims(img, axis = 0), batch_size=1, shuffle=False))
    
    class_weights = model.layers[-3].get_weights()[0] # weights from softmax layer
    
    get_output = K.function([model.layers[0].input], [model.get_layer(layer_name).output, model.layers[-3].output])
    [conv_outputs, predictions] = get_output(aug_img)

    conv_outputs = conv_outputs[0, :, :, :]
    class_idx = np.argmax(predictions)
    
    cam = np.zeros(dtype = np.float32, shape = conv_outputs.shape[0:2])
    
    for i, w in enumerate(class_weights[:, class_idx]):
        cam += w * conv_outputs[:, :, i]

    cam /= np.max(cam)
    cam = cv2.resize(cam, (width, height)) # the height and the width of the original image.
    
    heatmap = cv2.applyColorMap(np.uint8(255 * (255 - cam)), colormap)
    heatmap[np.where(cam < 0.2)] = 0   
    predictions = predictions[0]
    
    return predictions, heatmap
