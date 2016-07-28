#Softmax Regression
##1.Summary

    This algorithm is a normal situation of logistic regression. Logistic regression is used to classify 2 lables. 
    
    This algorithm is used to solve multi lables classification.

    In the logistic regression, we had a trainning set {(x1,y1),(x2,y2),...,(xm,ym)}, the labels y have 2 value 
    
    which are 1 and 0. In the softmax regression,  the labels y have k different values.
    
##2.Process
    

    As same as logistic regression, this algorithm has hypothesis function and cost function.
    
###(1) Hypothesis function
    
    The hypothesis function of softmax regression is considered to estimate the probability that p(y = j | x) 
    
    for each value of j = 1, ..., k. In other words, For some input x vector, It estimates the probability 
    
    of the k sorts different possible values. Thus, our hypothesis will output a k dimensional vector 
    
    (whose elements sum to 1) giving us our k estimated probabilities. 
    
    Concretely, our hypothesis hθ(x) takes the form:
    
<img src="http://chart.googleapis.com/chart?cht=tx&chl=h_%7B%5Ctheta%7D(x%5E%7B(i)%7D)%3D%5B%20p(y%5E%7B(i)%7D%3D1%7Cx%5E%7B(i)%7D%3B%5Ctheta)%2C%20p(y%5E%7B(i)%7D%3D2%7Cx%5E%7B(i)%7D%3B%5Ctheta)%2C...%20%2Cp(y%5E%7B(i)%7D%3Dk%7Cx%5E%7B(i)%7D%3B%5Ctheta)%5D%5E%7BT%7D" style="border:none;" />
    
<img src="http://chart.googleapis.com/chart?cht=tx&chl=%3D%5Cfrac%7B1%7D%7B%5Csum_%7Bj%3D1%7D%5Ek%20e%5E%7B%5Ctheta_%7Bj%7D%5E%7BT%7Dx%5E%7B(i)%7D%7D%7D%5B%0Ae%5E%7B%5Ctheta_%7B1%7D%5E%7BT%7Dx%5E%7B(i)%7D%7D%2C%5C%20%5C%20%5C%20%0Ae%5E%7B%5Ctheta_%7B2%7D%5E%7BT%7Dx%5E%7B(i)%7D%7D%2C%5C%20%5C%20%5C%20%0A.%5C%20%5C%20%5C%20.%20%5C%20%5C%20%5C.%5C%20%5C%20%5C%20%2C%0Ae%5E%7B%5Ctheta_%7Bk%7D%5E%7BT%7Dx%5E%7B(i)%7D%7D%0A%5D%5E%7BT%7D" style="border:none;" />

    Need to clear, Xi is n+1 dimension, the given Yi is k dimension, like [0,0,.1,0]. Theta is k*n+1.
    
###(2) Cost function

    Before learning cost function formula, the indicator function 1{.} is needed to understand.
    
    The rule is 1{a true statement} = 1, and 1{a false statement} = 0. 
    
    The cost function is given as below.
    
<img src="http://chart.googleapis.com/chart?cht=tx&chl=J(%5Ctheta)%3D-%5Cfrac%7B1%7D%7Bm%7D%5B%7B%5Csum_%7Bi%3D1%7D%5Em%20%5Csum_%7Bj%3D1%7D%5Ek%201%7By%5E%7B(i)%7D%3D1%7Dlog%5Cfrac%7Be%5E%7B%20%5Ctheta_%7Bj%7D%5E%7BT%7Dx%5E%7B(i)%7D%20%20%20%7D%7D%7B%20%5Csum_%7Bl%3D1%7D%5Ek%20e%5E%7B%20%5Ctheta_%7Bl%7D%5E%7BT%7Dx%5E%7B(i)%7D%7D%7D%5D" style="border:none;" />
    
    
    
    
    
    