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


##4.Gradient check

    This algorithm is really a useful method of gradient checking, which come from definition of derivative.
    
<img src="http://chart.googleapis.com/chart?cht=tx&chl=%5Cfrac%7Bd%7D%7Bd%5Ctheta%7DJ(%5Ctheta)%20%3D%20lim_%7B%5Cepsilon%20-%3E0%7D%0A%5Cfrac%7BJ(%5Ctheta%2B%5Cepsilon)-J(%5Ctheta-%5Cepsilon)%7D%7B2%5Cepsilon%7D" style="border:none;" />

    This is a iteration of gradient descent. 
    
<img src="http://chart.googleapis.com/chart?cht=tx&chl=%5Ctheta%3A%3D%5Ctheta%20-%20%5Calpha%5Cfrac%7Bd%7D%7Bd%5Ctheta%7DJ(%5Ctheta)" style="border:none;" />

    In the gradient descent method, g() is a function which approach to derivative of J()
    
<img src="http://chart.googleapis.com/chart?cht=tx&chl=%5Ctheta%3A%3D%5Ctheta%20-%20%5Calpha%20g(%5Ctheta)" style="border:none;" />

    The checking is to calculate g() is equal to about derivative of J() when epsilon is 0.0001.
    
    When theta is a vector, the below formula is given to calc theta+epsilon.
    
<img src="http://chart.googleapis.com/chart?cht=tx&chl=%5Ctheta%5E%7B(i%2B)%7D%20%3D%20%5Ctheta%20%2B%20%5Cepsilon%20%5Ctimes%20%5Cvec%7Be_%7Bi%7D%7D" style="border:none;" />

<img src="http://chart.googleapis.com/chart?cht=tx&chl=%5Cvec%7Be_%7Bi%7D%7D%20%3D%20%5B0%2C0%2C...%2C1%2C...%2C0%5D%5E%7BT%7D" style="border:none;" />
    
    The i element of e is 1, others are 0. g() is a vetor after looping for 0 to n, n is dimension of theta.

    Using it will give you self-confidence of your gradient descent method in any situation.


    
    

    
    
    
    
