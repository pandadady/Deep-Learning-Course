#Stacked Autoencoder

##1.Summary

    I am very excited to arrive this session which has a algorithm much more like deep learning network.
    
    This article is going to introduce stacked autoencoder from 6 aspect.
    
    (1) Advantages of deep networks over shallow networks 
    
    (2) The problems of traditional deep network.
    
    (3) Greedy layer-wise training
    
    (4) Stacked autoencoder classification
    
    (5) Fine-tuning 
    
    (6) Experiment
    
##2. Advantages of deep networks over shallow networks 
    
        The primary advantage is that it can compactly represent a significantly larger set of fuctions than 
    
    shallow networks. In other words, if a shallow network want to achieve the same performance , it must have much 
    
    more hidden layer nodes than a deep network. Thus, deep network is more efficient.
    
##3 The problems of traditional deep network

    Data probelm. Tranditional deep network need labeled data which is scarcity. If tranditional deep network don't 
    
    have enough input data ,Over fitting will happen.
    
    Local optima problem. Training a neural network using supervised learning involves solving a highly non-convex 
    
    optimization problem. In a deep network, this problem turns out to be rife with bad local optima, and training 
    
    with gradient descent (or methods like conjugate gradient and L-BFGS) no longer work well.
    
    Diffusion of gradients problem. when using backpropagation to compute the derivatives, the gradients that are 
    
    propagated backwards (from the output layer to the earlier layers of the network) rapidly diminish in magnitude 
    
    as the depth of the network increases. As a result, the derivative of the overall cost with respect to the 
    
    weights in the earlier layers is very small. Thus, when using gradient descent, the weights of the earlier 
    
    layers change slowly, and the earlier layers fail to learn much. This problem is often called the 
    
    "diffusion of gradients."
    
##4.Greedy layer-wise training


