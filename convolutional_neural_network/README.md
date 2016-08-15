#Convolutional Neural Network

##1.Summary

    This algorithm is usually used in the image recognition. It has excellent performance for large image processing.
    
    After 2 month learning, it's the time to learn the famous 'Feature extraction using convolution'.
    
##2.Process
    
    In the before articles, the sample is 8*8 or 28*28, either self-taught or sofmax algorithm work well, when the 
    
    sample become 96*96, the calculation is a challenage to the algorithms. The algorithm in this article is 
    
    designed to solve large image recognition problem.
    
###(1) Extract convolution features
    
    Assumed that the sample is 96*96, sparse autoencoder needs 10000 input nodes, assumed it is going to learn 
    
    100 features, so there is 100*10000 weights to learn. The learning process will be slow.
    
    The process of solving this problem is given as below
    
    1. Select randomly a small sample from large image,such as 8*8.
    
    2. Use a hidden unit (may have several hidden nodes) to extract sample features.
    
    3. Sample features do convolution with the original image to get convolutional features.
    
    4. Loop all the hidden units.
    
    Assumed there is 100 features needed to be learn, the size of large image is 96*96,  when using the old feature 
    
    extraction method , the weights number is 100*[100*100]. When using convolution features extraction method, 
    
    the weights number is 89*89*100, it is less than before but not enough. The next method will finish the problem.
    
###(2) Pooling 

    89*89*100 weights need a classification with 89*89*100 dimensions input nodes, and the algorithm is easy to be 
    
    over-fitting. 
    
    Pooling is a data aggregation technology.
    
    1.Split the convolved features matrix into several non coincidence areas.
    
    2.calc the max or average value of area.
    
    3.The calculation results compose a new matrix called pooled features matrix.
    
    Assume that area is 25*89*89, and according the rules given above, the pooled features matrix is 2*2
    
    Pooling technology will reduce input dimension into appropriate number.

##3.Exercise

    In this exercise you will use the features you learned on 8x8 patches sampled from images from the STL-10 dataset
    
    in the earlier exercise on linear decoders for classifying images from a reduced STL-10 dataset applying 
    
    convolution and pooling. The reduced STL-10 dataset comprises 64x64 images from 4 classes (airplane, car, cat, dog).
    
    
