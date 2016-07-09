#BP neural network

###1.Summary
    
    It is a classic neural network algorithm. Tutorials on this algorithm are everywhere.
    
    This page don't intent to explain in detail，but will explain the core related to algorithm achieve.
    
###2.Structure


<img style="-webkit-user-select: none; cursor: zoom-in;" src="http://ufldl.stanford.edu/wiki/images/3/3d/SingleNeuron.png" width="360" height="177">

    the simplest possible neural network, one which comprises a single "neuron."
    
    This "neuron" is a computational unit that takes as input x1,x2,x3 (and a +1 intercept term),
    
    and outputs is as follow. 
    
<img src="http://chart.googleapis.com/chart?cht=tx&chl=h_%7BW%2Cb%7D(x)%3Df(W%5E%7BT%7Dx)%3Df(w_%7B1%7Dx_%7B1%7D%2Bw_%7B2%7Dx_%7B2%7D%2Bw_%7B3%7Dx_%7B3%7D%2Bb)" style="border:none;" />

    f() is called activation function. Common activation function table is as follows
    
<img style="-webkit-user-select: none; cursor: zoom-in;" src="http://image95.360doc.com/DownloadImg/2016/03/1611/67845301_11.png">
<img style="-webkit-user-select: none; cursor: zoom-in;" src="http://image95.360doc.com/DownloadImg/2016/03/1611/67845301_12.png">
    
    For example, here is a small neural network:
    
<img style="-webkit-user-select: none" src="http://ufldl.stanford.edu/wiki/images/9/99/Network331.png" width="400" height="282">

###3.Characteristic

###4.Process
    
