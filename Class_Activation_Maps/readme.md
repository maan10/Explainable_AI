# **Class activation maps** #

<p align="justify">Class activation maps are used to localize the skin lesion of the most likely class for the input image predicted by the model. This method works with CNN architecture that uses Global Average Pooling (GAP) Layer.</p>  
  
To generate class activation maps, we perform the following two steps:  

1. <p align="justify">The softmax layer outputs probabilities for the seven classes. The argmax function over all the probabilities results in the most likely class for the image. The weights connected between the global average pooling layer and the softmax layer, i.e., the weights w<sub>1</sub>, w<sub>2</sub>,. . . , w<sub>n+1</sub> of that class are extracted. Next, the feature maps from the last convolution layer are extracted. Hence, for n feature maps before the global average pooling layer, we get n weights connecting between the GAP layer and the Softmax layer.</p>  
2. <p align="justify">Each feature map is multiplied with the corresponding weight, as shown in Figure below. The weight signifies the importance of the individual channel in the feature map. The final weighted sum is the heatmap of the class predicted. The size of the heatmap is the size of the feature map. The heatmap is first resized to the input image size, and then it is overlaid on top of the input to obtain results from CAM, as shown in Figure below.</p>
<div align="center">
<img src="https://github.com/rao208/Explainable_AI/blob/master/Images/CAM-1.svg" >
<p>Approach for Class Activation Maps</p>
</div>  

## **Results** ##
<p align="justify"> Figure 1 to Figure 7 represents the implementation of class activation maps on seven different classes given in the dataset. The top image in each figure’s subplot (a) represents the input image, i.e., skin lesion image of size (450, 600, 3). The correct label and the probability assigned to the correct label is written under each image. The bottom image in each figure’s subplot (a) is a bar plot, which denotes three labels considered most probable by the model. The green color represents the correct label, and the red color represents all the other labels in the bar plot. Subplot (b) represents the pre-processed image i.e. normalized in the range [-1, 1]. Subplot (c) represents the heatmap superimposed on the input image using Jet colormap array ranging from blue (low value) to red (high value). The red is the high-intensity area, i.e., the region is very important, and blue indicates the low-intensity area, i.e., the less important region, but still relevant enough to locate the skin lesion. As expected, the trained CNN model is looking for a specific feature. Based on the different morphologic variants such as rash, nodule, pigmentation, irregular patch, cyst on the surface of the skin, or areas of skin that look different from the surrounding area, CNN determines the coarse and irregular portions of a lesion to be important features. These features are used as the basis for its classification of skin lesions. If the predicted probability of the target class is high, then the heatmap on the important features, i.e., the affected area, is more intense. The higher the intensity of the heatmap, the model is surer about the location of the affected area based on these learned features. </p>
<div align="center">
<img src="https://github.com/rao208/Explainable_AI/blob/master/Class_Activation_Maps/Results/akk_5_cam-1.svg" >
<p><b>Figure 1:</b> Class Activation Maps result in Actinic Keratoses class: (a) Input Image with top 3 predictions (b) Pre-processed Image (c) Heatmap on the input image</p>
</div>  

<div align="center">
<img src="https://github.com/rao208/Explainable_AI/blob/master/Class_Activation_Maps/Results/bcc_1_cam-1.svg" >
<p><b>Figure 2:</b> Class Activation Maps result in Basal Cell Carcinoma class: (a) Input Image with top 3 predictions (b) Pre-processed Image (c) Heatmap on the input image</p>
</div>  

<div align="center">
<img src="https://github.com/rao208/Explainable_AI/blob/master/Class_Activation_Maps/Results/bkk_51_cam-1.svg" >
<p><b>Figure 3:</b> Class Activation Maps result in Benign Keratosis class: (a) Input Image with top 3 predictions (b) Pre-processed Image (c) Heatmap on the input image</p>
</div>  

<div align="center">
<img src="https://github.com/rao208/Explainable_AI/blob/master/Class_Activation_Maps/Results/df_9_cam.svg" >
<p><b>Figure 4:</b> Class Activation Maps result in Dermatofibroma class: (a) Input Image with top 3 predictions (b) Pre-processed Image (c) Heatmap on the input image</p>
</div>  

<div align="center">
<img src="https://github.com/rao208/Explainable_AI/blob/master/Class_Activation_Maps/Results/mcy_2_cam-1.svg" >
<p><b>Figure 5:</b> Class Activation Maps result in Melanocytic Nevi class: (a) Input Image with top 3 predictions (b) Pre-processed Image (c) Heatmap on the input image</p>
</div>  

<div align="center">
<img src="https://github.com/rao208/Explainable_AI/blob/master/Class_Activation_Maps/Results/mel_38_cam.svg" >
<p><b>Figure 6:</b> Class Activation Maps result in Melanoma class: (a) Input Image with top 3 predictions (b) Pre-processed Image (c) Heatmap on the input image</p>
</div>  

<div align="center">
<img src="https://github.com/rao208/Explainable_AI/blob/master/Class_Activation_Maps/Results/vasc_4_cam-1.svg" >
<p><b>Figure 7:</b> Class Activation Maps result in Vascular Lesions class: (a) Input Image with top 3 predictions (b) Pre-processed Image (c) Heatmap on the input image</p>
</div>  
