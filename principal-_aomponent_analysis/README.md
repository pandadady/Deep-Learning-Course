#Principal Component Analysis
##1.Summary

    Principal component analysis is a method of dimension reduction,It finds out a few synthetic 
    
    variables from many variables, which can represent the information of the original variables 
    
    as much as possible, and they are not related to each other.In mathematics, This data 
    
    transformation method is called linear transformation. There may be many sorts of linear transformation
    
    composation to achvieve transformation. Almong these,PCA wants to search a transformation along the 
    
    direction of maximum variance.
    
<img src="http://chart.googleapis.com/chart?cht=tx&chl=Var(F_%7B1%7D)%3EVar(F_%7B2%7D)%3E......%3EVar(F_%7Bp%7D)" style="border:none;" />
    
    F1 is the first principal component. F2 is the second principal component.
    
    Any two principal components are linearly independent
    
<img src="http://chart.googleapis.com/chart?cht=tx&chl=Cov(F_%7B1%7D%2CF_%7B2%7D)%20%3D%200" style="border:none;" />

    Transformation matrix is needed to follow the below formula.
    
<img src="http://chart.googleapis.com/chart?cht=tx&chl=AA%5E%7BT%7D%3DI" style="border:none;" />

##2.Derivation
    
    F is principal component matrix.
    
<img src="http://chart.googleapis.com/chart?cht=tx&chl=F%3D%5BF_%7B1%7D%2CF_%7B2%7D%2C...%2CF_%7Bp%7D%5D" style="border:none;" />

    A is transformation matrix.
    
<img src="http://chart.googleapis.com/chart?cht=tx&chl=A%3D%5BA_%7B1%7D%2CA_%7B2%7D%2C...%2CA_%7Bp%7D%5D" style="border:none;" />

    X is data matrix.
    
<img src="http://chart.googleapis.com/chart?cht=tx&chl=X%3D%5BX_%7B1%7D%2CX_%7B2%7D%2C...%2CX_%7Bp%7D%5D" style="border:none;" />

<img src="http://chart.googleapis.com/chart?cht=tx&chl=F%3DAX" style="border:none;" />
    
    Covariance matrix of F is 
    
<img src="http://chart.googleapis.com/chart?cht=tx&chl=Var(F)%3DVar(AX)%3DE%5B(AX-E(AX))(AX-E(AX))%5E%7BT%7D%5D" style="border:none;" />
    
<img src="http://chart.googleapis.com/chart?cht=tx&chl=%3DAE%5B(X-E(X))(X-E(X))%5E%7BT%7D%5DA%5E%7BT%7D" style="border:none;" />

    Covariance matrix of X is 
    
<img src="http://chart.googleapis.com/chart?cht=tx&chl=%5CSigma%20%3D%20E%5B(X-E(X))(X-E(X))%5E%7BT%7D%5D" style="border:none;" />

<img src="http://chart.googleapis.com/chart?cht=tx&chl=Var(F)%3DA%5CSigma%20A%5E%7BT%7D" style="border:none;" />

    Because Sigma matrix is p order matrix, We can have the following formula.
    
<img src="http://chart.googleapis.com/chart?cht=tx&chl=%5CSigma%20%3D%20U%20%5CLambda%20U%5E%7BT%7D%5C%5C%0A%0AUU%5E%7BT%7D%3DI%5C%5C%0A%0A%5CLambda%20%3D%20Diagonal%20%5C%20%5C%20%5C%20%5C%20matrix%5B%5Clambda_%7B1%7D%2C%5Clambda_%7B2%7D%2C...%2C%5Clambda_%7Bp%7D%5D" style="border:none;" />

    Covariance matrix of F is changed to
    
<img src="http://chart.googleapis.com/chart?cht=tx&chl=Var(F)%20%3D%20AU%20%5CLambda%20U%5E%7BT%7DA%5E%7BT%7D%5C%5C%0A%0A" style="border:none;" />

    When U=A', Covariance matrix of F have the max value.
    
<img src="http://chart.googleapis.com/chart?cht=tx&chl=Var(F)%20%3D%20%5CLambda%20%5C%5C%0A%5Clambda_%7B1%7D%3E%5Clambda_%7B2%7D%3E...%3E%5Clambda_%7Bp%7D%5C%5C%0AVar(F_%7B1%7D)%3EVar(F_%7B2%7D)%3E...%3EVar(F_%7Bp%7D)" style="border:none;" />
