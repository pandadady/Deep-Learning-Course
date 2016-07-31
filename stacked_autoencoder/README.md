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

    the main idea is to train the layers of the network one at a time, so that we first train a network with 1 hidden 
    
    layer(like self-taught learning), and only after that is done, train a network with 2 hidden layers, and so on. 
    
    Training is usually unsupervised.Fine-tune the entire network.
    
    Data probelm is solved by self-taught learning.
    
    Gradients does not need to be transmitted from the output layer to the first layer， so diffusion of gradient 
    
    problem is sloved.
    
##5.Stacked autoencoder classification    
    
    The structure is like below.
    
<img alt="Stacked Combined.png" src="http://ufldl.stanford.edu/wiki/images/5/5c/Stacked_Combined.png" width="500" height="434" /></a>

    The output of first layer is the input of second layer, and go on . First layer learns primary features.
    
    The second layer learns secondary features. The secondary features will add into softmax classification.
    
##6.Fine-tuning

    The process is almost backpropagation algorithm.
    
    (1) Feedforward process. calc the probabilities.
    
    (2) Calc error of output.
    
<img src="http://chart.googleapis.com/chart?cht=tx&chl=%5Cdelta%5E%7B(n_%7Bl%7D)%7D%3D-(%5Cnabla%20_%7B%5Calpha%5E%7Bn_%7Bl%7D%7D%7DJ)f'(z%5E%7B(n_%7Bl%7D)%7D)%5C%5C%0A%5Cnabla%20_%7B%5Calpha%5E%7Bn_%7Bl%7D%7D%7DJ%20%3D%20%5Ctheta%20%5E%7BT%7D(I-P)" style="border:none;" />
    
    (3) Calc error of other layers
    
<img src="http://chart.googleapis.com/chart?cht=tx&chl=%5Cdelta%5E%7B(l)%7D%3D((W%5E%7B(l)%7D)%5E%7BT%7D%5Cdelta%20%5E%7B(l%2B1)%7D)f'(z%5E%7B(l)%7D)" style="border:none;" />

    (4) Compute the desired partial derivatives
    
<img src="http://chart.googleapis.com/chart?cht=tx&chl=%5Cnabla%20W%5E%7B(l)%7D%20%3D%20%5Cdelta%20%5E%7B(l%2B1)%7D(%5Calpha%5E%7B(l)%7D)%5E%7BT%7D%5C%5C%0A%5Cnabla%20b%5E%7B(l)%7D%20%3D%20%5Cdelta%20%5E%7B(l%2B1)%7D" style="border:none;" />
