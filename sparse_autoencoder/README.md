#Sparse Autoencoder
##1.Summary

    An autoencoder neural network is an unsupervised learning algorithm that applies backpropagation, 
    
    setting the target values to be equal to the inputs.
    
##2.Structure

<img style="-webkit-user-select: none; cursor: zoom-in;" src="http://ufldl.stanford.edu/wiki/images/thumb/f/f9/Autoencoder636.png/400px-Autoencoder636.png" width="400" height="445">

    The output of algorithm equals to input. The hidden layer elements number is less then the number of input layer number.
    
    In fact, this simple autoencoder often ends up learning a low-dimensional representation very similar to PCAs.
    
    Sparsity means the none-zero elements number is nmuch more less than  other elements number.
    
    Sparse Autoencoder can use little number of weights to indicate the original vetor.
    
##3.Process
    
    Alpha_j is the activation value of the No.j hidden layer element. Introduce the concept of active value, here
    
    is the mean active value of all the training sample.
    
<img src="http://chart.googleapis.com/chart?cht=tx&chl=%5Chat%7B%5Crho_%7Bj%7D%20%7D%20%3D%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bi%3D1%7D%5Em%20%5B%5Calpha%5E%7B(2)%7D_%7Bj%7D(x%5E%7B(i)%7D)%5D" style="border:none;" />
    
    
    
