#Sparse Autoencoder
##1.Summary

    An autoencoder neural network is an unsupervised learning algorithm that applies backpropagation, 
    
    setting the target values to be equal to the inputs.
    
##2.Structure

<img style="-webkit-user-select: none; cursor: zoom-in;" src="http://ufldl.stanford.edu/wiki/images/thumb/f/f9/Autoencoder636.png/400px-Autoencoder636.png" width="400" height="445">

    The output of algorithm equals to input. The hidden layer elements number is less then the number of input 
    
    layer number.In fact, this simple autoencoder often ends up learning a low-dimensional representation very 
    
    similar to PCAs. Sparsity means the none-zero elements number is nmuch more less than  other elements number.
    
    Sparse Autoencoder can use little number of weights to indicate the original vetor.
    
##3.Process

    How to achieve sparsity？
    
    The core is to introduce the concept of active value.Alpha_j is the activation value of the No.j hidden layer 
    
    element.Here is the mean active value of all the training sample.
    
<img src="http://chart.googleapis.com/chart?cht=tx&chl=%5Chat%7B%5Crho_%7Bj%7D%20%7D%20%3D%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bi%3D1%7D%5Em%20%5B%5Calpha%5E%7B(2)%7D_%7Bj%7D(x%5E%7B(i)%7D)%5D" style="border:none;" />

    It is easy to think that the mean active value could be small in order to achieve sparsity weights.
    
    Sparse parameter rho is used to restrict the mean active value of all the training sample.
    
<img src="http://chart.googleapis.com/chart?cht=tx&chl=%5Chat%7B%5Crho%20_%7Bj%7D%7D%20%3D%20%5Crho" style="border:none;" />

    In order to achieve this restriction, KL formula is given as below.

<img src="http://chart.googleapis.com/chart?cht=tx&chl=KL%3D%5Cbeta%20%5B%5Csum_%7Bj%3D1%7D%5E%7Bnh%7D%20%5Crho%20log%5Cfrac%7B%5Crho%7D%7B%5Chat%7B%5Crho_%7Bj%7D%7D%7D%2B(1-%5Crho)log%7B%5Cfrac%7B1-%5Crho%7D%7B1-%5Chat%7B%5Crho_%7Bj%7D%7D%7D%5D" style="border:none;" />
    
    Add this restriction into cost function.
    
<img src="http://chart.googleapis.com/chart?cht=tx&chl=J_%7Bsparse%7D(W%2Cb)%3DJ(W%2Cb)%2BKL" style="border:none;" />

    The following step is to get the min value of cost function. Error is propagating from back to front.

    But the error formula has little different to the bpNN algorithm.

<img src="http://chart.googleapis.com/chart?cht=tx&chl=%5Cdelta%20_%7Bi%7D%5E%7B(2)%7D%3D%5B(%5Csum_%7Bj%3D1%7D%5E%7Bnh%7DW_%7Bij%7D%5E%7B(2)%7D%20%5Cdelta%20_%7Bi%7D%5E%7B(3)%7D)%2B%5Cbeta%20(-%5Cfrac%7B%5Crho%7D%7B%5Chat%7B%5Crho_%7Bi%7D%7D%7D%2B%5Cfrac%7B1-%5Crho%7D%7B1-%5Chat%7B%5Crho_%7Bi%7D%7D%7D)%5Df'(%5Calpha_%7Bi%7D%5E%7B(2)%7D)" style="border:none;" />



    

    
    
    
    
