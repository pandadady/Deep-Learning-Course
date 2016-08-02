#Linear Decoders

##1.Summary

     In our previous description of autoencoders (and of neural networks), every neuron in the neural 
     
     network used the same activation function(sigmod). Because we used a sigmoid activation function for f(),
     
     we needed to constrain or scale the inputs to be in the range [0,1], since the sigmoid function outputs 
     
     numbers in the range [0,1]. While some datasets like MNIST fit well with this scaling of the output, 
     
     this can sometimes be awkward to satisfy. For example, if one uses PCA whitening, the input is no longer 
     
     constrained to [0,1] .
     
     Linear decoder is designed to solved this problem.
     
##2.Process

    The output layer activation function use this formula, called linear activation function.
    
<img src="http://chart.googleapis.com/chart?cht=tx&chl=f(z)%5C%20%5C%20%5C%20%20%3D%20%20%5C%20%5C%20%5C%20z%20" style="border:none;" />

   In the hidden layer of the network, we still use a sigmoid activation function.
   
   An autoencoder in this configuration--with a sigmoid (or tanh) hidden layer and a linear output layer--is 
   
   called a linear decoder.
  
<img src="http://chart.googleapis.com/chart?cht=tx&chl=%5Chat%7Bx%7D%20%3D%20%5Calpha%5E%7B(3)%7D%3Dz%5E%7B(3)%7D%20%3D%20W%5E%7B(2)%7D%5Calpha%2Bb%5E%7B(2)%7D%5C%5C%0A%0A%5Cdelta%20%5E%7B(3)%7D%3D-(y_%7Bi%7D-%5Chat%7Bx_%7Bi%7D%7D)f'_%7B3%7D(z_%7Bi%7D%5E%7B(3)%7D)%5C%5C" style="border:none;" />

    beacause:
    
<img src="http://chart.googleapis.com/chart?cht=tx&chl=f'_%7B3%7D(z_%7Bi%7D%5E%7B(3)%7D)%20%3D%201%5C%5C%0A%0A%5Cdelta%20%5E%7B(3)%7D%3D-(y_%7Bi%7D-%5Chat%7Bx_%7Bi%7D%7D)" style="border:none;" />

<img src="http://chart.googleapis.com/chart?cht=tx&chl=%5Cdelta%20%5E%7B(2)%7D%3D((W%5E%7B(2)%7D)%5E%7BT%7D%5Cdelta%20%5E%7B(3)%7D)%5Ccdot%20f'_%7B2%7D(z%5E%7B(2)%7D)" style="border:none;" />
   
