### **Computer Vision Weather Forecasting 🌤️🌧️**



##### Project Overview

Designed and optimized a computer vision pipeline to accurately classify real-world weather patterns from imagery. This project documents the complete machine learning lifecycle, from an initial custom-built Convolutional Neural Network (CNN) to a production-ready Transfer Learning architecture.



##### The Challenge \& The Pivot

Initially, I designed and trained a custom CNN from scratch on a curated dataset of 1,125 images across 4 distinct weather classes (Cloudy, Rain, Shine, Sunrise). However, due to the compact size of the dataset, the 11-million parameter model encountered severe overfitting—memorizing the training data but failing to generalize to unseen validation images (peaking at \~37% validation accuracy).



To achieve production-grade metrics without artificially inflating the dataset, I pivoted to Transfer Learning using Microsoft's pre-trained ResNet50 architecture.



##### The Solution

By utilizing ResNet50's powerful pre-trained feature extraction maps (trained on ImageNet) and freezing the base layers, the model no longer had to learn basic geometric shapes from scratch.



Combined with aggressive real-time Data Augmentation and Global Average Pooling, the optimized model successfully broke the 90% validation accuracy barrier, proving highly robust on unseen data.



##### Repository Structure



* /notebooks: Contains the Jupyter/Colab notebooks detailing the initial custom CNN experiment and the final ResNet50 training pipeline.



* /models: Contains the serialized .h5 model ready for web/API deployment.



* /data: Sample imagery of the 4 weather classes used for validation.



##### Technical Stack \& Architecture



* Frameworks: TensorFlow, Keras, NumPy, Matplotlib



* Base Model: ResNet50 (ImageNet weights)



* Optimization: Adam Optimizer, Sparse Categorical Cross-Entropy



* Anti-Overfitting Techniques: 

&#x09;	\* Global Average Pooling 2D (reduced parameter bottleneck)



&#x09;	\* Deep Dropout Layers (up to 50%)



&#x09;	\* Dynamic Data Augmentation (Random Rotation, Zoom, and Horizontal Flipping)

