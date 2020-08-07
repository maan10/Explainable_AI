# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 03:14:52 2020

@author: Vanditha Rao
"""

from tensorflow.keras import backend as K
import numpy as np
import cv2

def visualize_cam(model, image):
    
    _, width, height, c = image.shape
    
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    final_conv_layer = layer_dict["conv_pw_13_relu"]

    class_weights = model.layers[-1].get_weights()[0]
    # print('class_weights', class_weights.shape)
    
    get_output = K.function([model.layers[0].input], [final_conv_layer.output, model.layers[-1].output])
    [conv_outputs, predictions] = get_output(image)

    conv_outputs = conv_outputs[0, :, :, :]
    class_idx = np.argmax(predictions)
    
    cam = np.zeros(dtype = np.float32, shape = conv_outputs.shape[0:2])
    
    for i, w in enumerate(class_weights[:, class_idx]):
        cam += w * conv_outputs[:, :, i]

    cam /= np.max(cam)
    
    # print('cam size', cam.shape)
    cam = cv2.resize(cam, (600, 450))
    
    heatmap = cv2.applyColorMap(np.uint8(255 * (255 - cam)), cv2.COLORMAP_JET)
    heatmap[np.where(cam < 0.2)] = 0   
    predictions = predictions[0]
    
    return predictions, heatmap