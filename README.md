# Table of contents
1. [Introduction](#Introduction)
    1. [Problem Statement](#Problem_Statement)
    2. [Objective](#Objective)
2. [Roadmap](#Roadmap)
3. [Contributing](#Contributing)
4. [Licence](#Licence)
5. [Link](#Link)

# **Introduction** #
<p align="justify">According to the Global Cancer Statistics 2018, cancer is a prominent cause of death worldwide, with nearly 9.6 million deaths in 2018 [1]. Non-melanoma skin cancer is the fifth most common cancer, with 1.04 million new cases in 2018 [1]. In contrast, Melanoma, the deadliest type of skin cancer, is the 19th most commonly occurring cancer with nearly 300,000 new cases in 2018 [1]. The early detection of cancer significantly improves the diagnosis of patients.</p>  
  
![alt text](<https://github.com/rao208/Explainable_AI/blob/master/Images/Cancer_Statistics_2018%20(1)-1.svg>)  

<p align="justify">Skin cancer is initially identified visually with an initial clinical screening, followed by dermatoscopy analysis, a biopsy, and histopathological examination. These tests consume time, and the patient may advance to a later stage of skin cancer. Over the last few years, Computer-Aided Diagnosis (CAD) system based on deep learning is proposed to deal with the detection of skin cancer. Automated lesion classification can both support physicians in their daily clinical routine and enable fast and cheap access to lifesaving diagnoses.</p>  
  
<p align="justify">CNN is the state-of-the-art for the image classification task and has provided a promising way to make the diagnosis process more efficient. It is achieving notable results in the classification of skin cancer. Esteva et al. [2] made the first advancement in skin cancer classification using the pre-trained GoogLeNet Inception V3 [3] CNN model. The reported accuracy of classification is 72.1 % ± 0.9 % on 129,450 clinical skin cancer images, including 3,374 dermoscopic images. In 2016, Yu et al.[4] came up with a CNN model with over 50 layers. The author used the International Symposium of Biomedical Imaging (ISBI) 2016 challenge dataset for the classification of malignant melanoma cancer. The best classification accuracy reported in this challenge was 85.5%. In 2018, Haenssle et al.[5] utilized a deep CNN to classify a binary diagnostic category of dermatoscopy melanocytic images and reported 88.9% ± 9.6% sensitivity and 75.7% ± 11.7% specificity for lesion classification. Dorj et al.[6] developed a multi-class classification using an Error-Correcting Output Codes (ECOC) Support vector machine (SVM) with pre-trained AlexNet Deep Learning CNN. The author reported an average accuracy of 95.1% (Squamous Cell Carcinoma), average sensitivity 98.9% (Actinic Keratosis), and specificity of 94.17% (Squamous Cell Carcinoma) in this work.</p>  

## **Problem Statement** ##
<p align="justify">Despite achieving remarkable results in terms of accuracy and their success in the classification task, CAD using deep learning has not reached a significant deployment in clinics and hospitals. The lack of deployment is because automatically learning the features comes with the lack of explanation in Artificial Intelligence (AI) driven systems, which means that they are a black-box. CNN does not offer a simple and understandable explanation because of the complex architecture with several to many layers and requires training of hundreds of thousands of parameters. In the medical domain, the treatment based on the diagnosis made should be established on an informed decision and not on the classification produced by a black-box model. A clinician would require more explanations than just the model’s predictions. Instead of concentrating only on the final accuracy, we also need to focus on providing explanations to the end users because they want to know why the model is predicting a particular class. Therefore, CNN must explain the rationale behind decisions so that it can be used in clinics and hospitals. <b>In general, a CAD system that supports the diagnosis of the clinicians and provides insights and visual explanations is more useful and desirable.</b></p>  
  
## **Objective** ## 
<p align="justify">The goal is to explore the presented research gap on how to explain the classification decisions of different types of skin lesions made by the CNN model trained on Human Against Machine with 10000 training images(HAM10000) skin lesion dataset. To the best of the author’s knowledge, very few research exists in skin cancer research that looked into the explanation of a CNN model trained on skin cancer datasets [7]. The aim is to visualize these explanations by generating heatmaps which are superimposed on the input image. The heatmaps hold great potential in assisting and understanding diagnostic decisions performed by CNNs.</p>

<p align="justify">Two explainable models are selected to test. The first method is the perturbationbased forward propagation method in which the input is perturbed, and its possible effects on the prediction of the model are examined. The perturbation based method used here is Occlusion Sensitivity. The second method is Class Activation Maps (CAM). The important regions of the image are highlighted by projecting back the weights of the output on the convolutional feature maps using this method. These are a good choice as they provide easy to understand heatmaps from the input images. The goal of the XAI method selection was to have different explaining models to have a good understanding of different approaches. Occlusion Sensitivity is meant to be used with all classification methods, whereas CAM can be used with layered models such as CNN. With two different methods, it is also sufficient to learn the differences, pros, and cons of each method. </p>  

# **Roadmap** #

- [ ] Additional Methods
    - [ ] Layerwise Relevance Propagation
    - [ ] DeepLift
    - [ ] Q&A for explanation
- [ ] Explanation to distinguish between cancerous and non-cancerous 

# **Contributing** #

Contributions are welcome on this repository! There are multiple ways to give a hand on this repository:

* Tackle new features from the roadmap
* Fix Typos
* Improve code quality
* Code coverage
* Documentation

# **Licence** #
MIT License  

# **Link** #  

* [Issue Tracker](https://github.com/rao208/Explainable_AI/issues)
* [Source Code](https://github.com/rao208/Explainable_AI)
