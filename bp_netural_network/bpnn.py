#!/usr/bin/python
# -*- coding:  utf-8 -*
"""
This module is used to learn bpnn algorithm
Authors: shiduo(shiduo@baidu.com)
Date:2016-7-15
"""
import random
import numpy
import math
import time
import scipy.optimize

class bpnn(object):
    """ The backward net class """
    
    def __init__(self, visible_size, hidden_size, output_size,lamda):
        """ Initialize parameters of the bpnn object """
        self.output_size = output_size    # number of output units
        self.visible_size = visible_size    # number of input units
        self.hidden_size = hidden_size      # number of hidden units
        self.lamda = lamda
        """ Set limits for accessing 'theta' values """
        
        self.limit0 = 0
        self.limit1 = hidden_size * visible_size
        self.limit2 = hidden_size * visible_size + output_size * visible_size
        self.limit3 = hidden_size * visible_size + output_size * visible_size + hidden_size
        self.limit4 = hidden_size * visible_size + output_size * visible_size + hidden_size + output_size
        
        """ Initialize Neural Network weights randomly
            W1, W2 values are chosen in the range [-r, r] """
        
        r = math.sqrt(6) / math.sqrt(visible_size + hidden_size + 1)
        
        rand = numpy.random.RandomState(int(time.time()))
        
        W1 = numpy.asarray(rand.uniform(low = -r, high = r, size = (hidden_size, visible_size)))
        W2 = numpy.asarray(rand.uniform(low = -r, high = r, size = (output_size, hidden_size)))
        
        """ Bias values are initialized to zero """
        
        b1 = numpy.zeros((hidden_size, 1))
        b2 = numpy.zeros((output_size, 1))

        """ Create 'theta' by unrolling W1, W2, b1, b2 """

        self.theta = numpy.concatenate((W1.flatten(), W2.flatten(),
                                        b1.flatten(), b2.flatten()))

    def sigmoid(self, x):
        """ Return elementwise sigmoid output of input array """
        return (1 / (1 + numpy.exp(-x)))
    
    def forward_process(self,input, W1,W2,b1,b2):
        """ forward propagation process """
        hidden_layer = self.sigmoid(numpy.dot(W1, input) + b1)
        output_layer = self.sigmoid(numpy.dot(W2, hidden_layer) + b2)
        return hidden_layer,output_layer
    
    def test_accuracy(self,input,output,W1,W2,b1,b2):
        """ Return accuracy of algorithm"""
        hidden_layer,output_layer = self.forward_process(input,W1,W2,b1,b2)
        n = output_layer.shape[1]
        right=0
        for i in range(n):
            if output_layer[0,i] < 0.2: p = 0
            else:p = 1
            if p == output[0,i]: right += 1
        accuracy = right*100.00/n
        saccuracy = str(accuracy)+"%"
#         errordist = scipy.spatial.distance.euclidean(output,output_layer)
        return saccuracy
    
    def bpCost(self, theta, input,output):
        """ Returns the cost of the bpnn and gradient at a particular 'theta' """
        """ Extract weights and biases from 'theta' input """
        W1 = theta[self.limit0 : self.limit1].reshape(self.hidden_size, self.visible_size)
        W2 = theta[self.limit1 : self.limit2].reshape(self.output_size, self.hidden_size)
        b1 = theta[self.limit2 : self.limit3].reshape(self.hidden_size, 1)
        b2 = theta[self.limit3 : self.limit4].reshape(self.output_size, 1)
        
        """ Compute output layers by performing a forward pass
            Computation is done for all the training inputs simultaneously """
        hidden_layer,output_layer=self.forward_process(input,W1,W2,b1,b2)
        
        """ Compute intermediate difference values using Backpropagation algorithm """
        diff = output_layer - output
        
        sum_of_squares_error = 0.5 * numpy.sum(numpy.multiply(diff, diff)) / input.shape[1]
        weight_decay         = 0.5 * self.lamda * (numpy.sum(numpy.multiply(W1, W1)) +
                                                   numpy.sum(numpy.multiply(W2, W2)))
        cost                 = sum_of_squares_error + weight_decay 
        del_out = numpy.multiply(diff, numpy.multiply(output_layer, 1 - output_layer))
        del_hid = numpy.multiply(numpy.dot(W2.T, del_out),numpy.multiply(hidden_layer, 1 - hidden_layer))
        
        """ Compute the gradient values by averaging partial derivatives
            Partial derivatives are averaged over all training examples """
        W1_grad = numpy.dot(del_hid, numpy.transpose(input))
        W2_grad = numpy.dot(del_out, numpy.transpose(hidden_layer))
        b1_grad = numpy.sum(del_hid, axis = 1)
        b2_grad = numpy.sum(del_out, axis = 1)
        
        W1_grad = W1_grad / input.shape[1] + self.lamda * W1
        W2_grad = W2_grad / input.shape[1] + self.lamda * W2
        b1_grad = b1_grad / input.shape[1]
        b2_grad = b2_grad / input.shape[1]
        """ Transform numpy matrices into arrays """
        
        W1_grad = numpy.array(W1_grad)
        W2_grad = numpy.array(W2_grad)
        b1_grad = numpy.array(b1_grad)
        b2_grad = numpy.array(b2_grad)
        
        """ Unroll the gradient values and return as 'theta' gradient """
        
        theta_grad = numpy.concatenate((W1_grad.flatten(), W2_grad.flatten(),
                                        b1_grad.flatten(), b2_grad.flatten()))
        print self.test_accuracy(input, output, W1, W2, b1, b2)
        return [cost, theta_grad]


def normalizeDataset(dataset):
    """ Normalize the dataset provided as input """
    dataset = dataset - numpy.mean(dataset)
    std_dev = 3 * numpy.std(dataset)
    dataset = numpy.maximum(numpy.minimum(dataset, std_dev), -std_dev) / std_dev
    dataset = (dataset + 1) * 0.4 + 0.1
    return dataset


def load_dataset(num_patches,patch_size):
    """ Create dataset randomly """
    dataset = numpy.zeros((num_patches,patch_size))
    for i in range(num_patches):
        dataset[i,] = numpy.random.randint(0,10,(1, patch_size)).flatten()
    labels = numpy.zeros((num_patches,1))
    for i in range(num_patches):
        labels[i,] = random.randint(0,1)
    return dataset.T,labels.T


def execute_bpnn():
    """ Loads data, trains the bpnn and visualizes the learned weights """
    lamda          = 0.0001 # weight decay parameter
    num_patches    = 10   # number of training examples
    max_iterations = 400  # number of optimization iterations
    visible_size = 2  # number of input units
    hidden_size  = 2  # number of hidden units
    output_size  = 1  # number of hidden units
    training_data,labels = load_dataset(num_patches, visible_size)
    encoder = bpnn(visible_size, hidden_size, output_size,lamda)
    opt_solution  = scipy.optimize.minimize(encoder.bpCost, encoder.theta, 
                                            args = (training_data,labels), method = 'BFGS', 
                                            jac = True, options = {'maxiter': max_iterations})
    
    
if __name__ == '__main__':
    
    execute_bpnn()
