#Softmax Regression
##1.Summary

    This algorithm is a normal situation of logistic regression. Logistic regression is used to classify 2 lables. 
    
    This algorithm is used to solve multi lables classification.

    In the logistic regression, we had a trainning set {(x1,y1),(x2,y2),...,(xm,ym)}, the labels y have 2 value 
    
    which are 1 and 0. In the softmax regression,  the labels y have k different values.
    
##2.Process
    
    As same as logistic regression, this algorithm has hypothesis function and cost function.
    
    The hypothesis function of softmax regression is considered to estimate the probability that p(y = j | x) 
    
    for each value of j = 1, ..., k. In other words, For some input x vector, It estimates the probability 
    
    of the k sorts different possible values. Thus, our hypothesis will output a k dimensional vector 
    
    (whose elements sum to 1) giving us our k estimated probabilities. 
    
    Concretely, our hypothesis hθ(x) takes the form:
    
<img src="http://chart.googleapis.com/chart?cht=tx&chl=h_%7B%5Ctheta%7D(x%5E%7B(i)%7D)%3D%5B%20p(y%5E%7B(i)%7D%3D1%7Cx%5E%7B(i)%7D%3B%5Ctheta)%2C%20p(y%5E%7B(i)%7D%3D2%7Cx%5E%7B(i)%7D%3B%5Ctheta)%2C...%20%2Cp(y%5E%7B(i)%7D%3Dk%7Cx%5E%7B(i)%7D%3B%5Ctheta)%5D%5E%7BT%7D" style="border:none;" />
    
    
    
