# -*- coding:  utf-8 -*
"""
This module is used to learn stacked autoencoder
Authors: shiduo
Date:2016年8月1日
"""
import math
import matplotlib.pyplot as plt
import struct
import numpy
import array
import time
import scipy.sparse
import scipy.optimize




class SoftmaxRegression(object):
    """ The Softmax Regression class """
    """ Initialization of Regressor object """

    def __init__(self, input_size, num_classes, lamda):
    
        """ Initialize parameters of the Regressor object """
    
        self.input_size  = input_size  # input vector size
        self.num_classes = num_classes # number of classes
        self.lamda       = lamda       # weight decay parameter
        
        """ Randomly initialize the class weights """
        
        rand = numpy.random.RandomState(int(time.time()))
        
        self.theta = 0.005 * numpy.asarray(rand.normal(size = (num_classes*input_size, 1)))
    
    
        
    def getGroundTruth(self, labels):
        """ Returns the groundtruth matrix for a set of labels """
    
        """ Prepare data needed to construct groundtruth matrix """
    
        labels = numpy.array(labels).flatten()
        data   = numpy.ones(len(labels))
        indptr = numpy.arange(len(labels)+1)
        
        """ Compute the groundtruth matrix and return """
        
        ground_truth = scipy.sparse.csr_matrix((data, labels, indptr))
        
        ground_truth = numpy.transpose(ground_truth.todense())
        
        return ground_truth
        
    
        
    def softmaxCost(self, theta, input, labels):
        """ Returns the cost and gradient of 'theta' at a particular 'theta' """
        """ Compute the groundtruth matrix """
    
        ground_truth = self.getGroundTruth(labels)
        """ Reshape 'theta' for ease of computation """
        
        theta = theta.reshape(self.num_classes, self.input_size)
        
        """ Compute the class probabilities for each example """
        
        theta_x       = numpy.dot(theta, input)
        hypothesis    = numpy.exp(theta_x)      
        probabilities = hypothesis / numpy.sum(hypothesis, axis = 0)
        
        """ Compute the traditional cost term """
        
        cost_examples    = numpy.multiply(ground_truth, numpy.log(probabilities))
        traditional_cost = -(numpy.sum(cost_examples) / input.shape[1])
        
        """ Compute the weight decay term """
        
        theta_squared = numpy.multiply(theta, theta)
        weight_decay  = 0.5 * self.lamda * numpy.sum(theta_squared)
        
        """ Add both terms to get the cost """
        
        cost = traditional_cost + weight_decay
        
        """ Compute and unroll 'theta' gradient """
        
        theta_grad = -numpy.dot(ground_truth - probabilities, numpy.transpose(input))
        theta_grad = theta_grad / input.shape[1] + self.lamda * theta
        theta_grad = numpy.array(theta_grad)
        theta_grad = theta_grad.flatten()
#         print 'cost',cost
#         print 'theta_grad',theta_grad 
        return [cost, theta_grad]
    
    
            
    def softmaxPredict(self, theta, input):
        """ Returns predicted classes for a set of inputs """
    
        """ Reshape 'theta' for ease of computation """
    
        theta = theta.reshape(self.num_classes, self.input_size)
        
        """ Compute the class probabilities for each example """
        
        theta_x       = numpy.dot(theta, input)
        hypothesis    = numpy.exp(theta_x)      
        probabilities = hypothesis / numpy.sum(hypothesis, axis = 0)
        
        """ Give the predictions based on probability values """
        
        predictions = numpy.zeros((input.shape[1], 1))
        predictions[:, 0] = numpy.argmax(probabilities, axis = 0)
        
        return predictions

class SparseAutoencoder(object):
    """ The Sparse Autoencoder class """
    """ Initialization of Autoencoder object """

    def __init__(self, visible_size, hidden_size, rho, lamda, beta):
    
        """ Initialize parameters of the Autoencoder object """
    
        self.visible_size = visible_size    # number of input units
        self.hidden_size = hidden_size      # number of hidden units
        self.rho = rho                      # desired average activation of hidden units
        self.lamda = lamda                  # weight decay parameter
        self.beta = beta                    # weight of sparsity penalty term
        
        """ Set limits for accessing 'theta' values """
        
        self.limit0 = 0
        self.limit1 = hidden_size * visible_size
        self.limit2 = 2 * hidden_size * visible_size
        self.limit3 = 2 * hidden_size * visible_size + hidden_size
        self.limit4 = 2 * hidden_size * visible_size + hidden_size + visible_size
        
        """ Initialize Neural Network weights randomly
            W1, W2 values are chosen in the range [-r, r] """
        
        r = math.sqrt(6) / math.sqrt(visible_size + hidden_size + 1)
        
        rand = numpy.random.RandomState(int(time.time()))
        
        W1 = numpy.asarray(rand.uniform(low = -r, high = r, size = (hidden_size, visible_size)))
        W2 = numpy.asarray(rand.uniform(low = -r, high = r, size = (visible_size, hidden_size)))
        
        """ Bias values are initialized to zero """
        
        b1 = numpy.zeros((hidden_size, 1))
        b2 = numpy.zeros((visible_size, 1))

        """ Create 'theta' by unrolling W1, W2, b1, b2 """

        self.theta = numpy.concatenate((W1.flatten(), W2.flatten(),
                                        b1.flatten(), b2.flatten()))

    
    
    def sigmoid(self, x):
        """ Returns elementwise sigmoid output of input array """
    
        return (1 / (1 + numpy.exp(-x)))

    def calc_hiddenlayer(self,theta,dataset):
        
        W1 = theta[self.limit0 : self.limit1].reshape(self.hidden_size, self.visible_size)
        W2 = theta[self.limit1 : self.limit2].reshape(self.visible_size, self.hidden_size)
        b1 = theta[self.limit2 : self.limit3].reshape(self.hidden_size, 1)
        b2 = theta[self.limit3 : self.limit4].reshape(self.visible_size, 1)
        hidden_layer = self.sigmoid(numpy.dot(W1, dataset) + b1)
        return hidden_layer
    
    def sparseAutoencoderCost(self, theta, input):
        """ Returns the cost of the Autoencoder and gradient at a particular 'theta' """
        """ Extract weights and biases from 'theta' input """
        W1 = theta[self.limit0 : self.limit1].reshape(self.hidden_size, self.visible_size)
        W2 = theta[self.limit1 : self.limit2].reshape(self.visible_size, self.hidden_size)
        b1 = theta[self.limit2 : self.limit3].reshape(self.hidden_size, 1)
        b2 = theta[self.limit3 : self.limit4].reshape(self.visible_size, 1)
#         print 'b1',b1.shape
#         print 'numpy.dot(W1, input)',numpy.dot(W1, input).shape
#         print 'numpy.dot(W1, input) + b1',(numpy.dot(W1, input) + b1).shape
        """ Compute output layers by performing a feedforward pass
            Computation is done for all the training inputs simultaneously """
        hidden_layer = self.sigmoid(numpy.dot(W1, input) + b1)
        output_layer = self.sigmoid(numpy.dot(W2, hidden_layer) + b2)
        
        """ Estimate the average activation value of the hidden layers """
        
        rho_cap = numpy.sum(hidden_layer, axis = 1) / input.shape[1]
        
        """ Compute intermediate difference values using Backpropagation algorithm """
        
        diff = output_layer - input
        
        sum_of_squares_error = 0.5 * numpy.sum(numpy.multiply(diff, diff)) / input.shape[1]
        weight_decay         = 0.5 * self.lamda * (numpy.sum(numpy.multiply(W1, W1)) +
                                                   numpy.sum(numpy.multiply(W2, W2)))
        KL_divergence        = self.beta * numpy.sum(self.rho * numpy.log(self.rho / rho_cap) +
                                (1 - self.rho) * numpy.log((1 - self.rho) / (1 - rho_cap)))
        cost                 = sum_of_squares_error + weight_decay + KL_divergence
        
        KL_div_grad = self.beta * (-(self.rho / rho_cap) + ((1 - self.rho) / (1 - rho_cap)))
        
        del_out = numpy.multiply(diff, numpy.multiply(output_layer, 1 - output_layer))
        del_hid = numpy.multiply(numpy.dot(numpy.transpose(W2), del_out) + \
                                 numpy.transpose(numpy.matrix(KL_div_grad)), \
                                 numpy.multiply(hidden_layer, 1 - hidden_layer))
        
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
        return [cost, theta_grad]
class BackPropagation (object):
    """ The BackPropagation class """
    def __init__(self, \
                 input_size, hidden_size_1, hidden_size_2, \
                 num_classes, theta1, theta2,theta3, lamda):
        self.theta = None
        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.num_classes = num_classes
        self.lamda = lamda
        
        self.limit0 = 0
        self.limit1 = self.limit0 + hidden_size_1 * input_size
        self.limit2 = self.limit1 + hidden_size_2 * hidden_size_1
        self.limit3 = self.limit2 + hidden_size_1
        self.limit4 = self.limit3 + hidden_size_2
        self.limit5 = self.limit4 + num_classes * hidden_size_2
        
        W1,b1 = self.get_w_b(theta1, hidden_size_1, input_size)
        W2,b2 = self.get_w_b(theta2, hidden_size_2, hidden_size_1)
        
        self.theta = numpy.concatenate((W1.flatten(), W2.flatten(),
                                        b1.flatten(), b2.flatten(), theta3.flatten()))
    def getGroundTruth(self, labels):
        """ Returns the groundtruth matrix for a set of labels """
    
        """ Prepare data needed to construct groundtruth matrix """
    
        labels = numpy.array(labels).flatten()
        data   = numpy.ones(len(labels))
        indptr = numpy.arange(len(labels)+1)
        
        """ Compute the groundtruth matrix and return """
        
        ground_truth = scipy.sparse.csr_matrix((data, labels, indptr))
        
        ground_truth = numpy.transpose(ground_truth.todense())
        
        return ground_truth  
     
    def get_w_b(self,theta,hidden_size,visible_size):
        limit0 = 0
        limit1 = hidden_size * visible_size
        limit2 = 2 * hidden_size * visible_size
        limit3 = 2 * hidden_size * visible_size + hidden_size
        limit4 = 2 * hidden_size * visible_size + hidden_size + visible_size
        W1 = theta[limit0 : limit1].reshape(hidden_size, visible_size)
        b1 = theta[limit2 : limit3].reshape(hidden_size, 1)

        return W1,b1
    def sigmoid(self, x):
        """ Returns elementwise sigmoid output of input array """
        return (1 / (1 + numpy.exp(-x)))
    def softmaxPredict(self, theta, input):
        """ Returns predicted classes for a set of inputs """
        """ Reshape 'theta' for ease of computation """
    
        """ Compute the class probabilities for each example """
        theta_x       = numpy.dot(theta, input)
        hypothesis    = numpy.exp(theta_x)      
        probabilities = hypothesis / numpy.sum(hypothesis, axis = 0)
        
        """ Give the predictions based on probability values """
        predictions = numpy.zeros((input.shape[1], 1))
        predictions[:, 0] = numpy.argmax(probabilities, axis = 0)
        return predictions
    def bpPredict(self,theta, input):
        W1 = theta[self.limit0 : self.limit1].reshape(self.hidden_size_1, self.input_size)
        W2 = theta[self.limit1 : self.limit2].reshape(self.hidden_size_2, self.hidden_size_1)
        b1 = theta[self.limit2 : self.limit3].reshape(self.hidden_size_1, 1)
        b2 = theta[self.limit3 : self.limit4].reshape(self.hidden_size_2, 1)
        theta3 = theta[self.limit4 : self.limit5].reshape(self.num_classes, self.hidden_size_2)
        L1_features = self.sigmoid(numpy.dot(W1, input) + b1)
        L2_features = self.sigmoid(numpy.dot(W2, L1_features) + b2)
        predictions = self.softmaxPredict(theta3, L2_features)
        return predictions
    def backPropagationCost(self, theta, input, output):
        W1 = theta[self.limit0 : self.limit1].reshape(self.hidden_size_1, self.input_size)
        W2 = theta[self.limit1 : self.limit2].reshape(self.hidden_size_2, self.hidden_size_1)
        b1 = theta[self.limit2 : self.limit3].reshape(self.hidden_size_1, 1)
        b2 = theta[self.limit3 : self.limit4].reshape(self.hidden_size_2, 1)
        theta3 = theta[self.limit4 : self.limit5].reshape(self.num_classes, self.hidden_size_2)
        
        L1_features = self.sigmoid(numpy.dot(W1, input) + b1)
        L2_features = self.sigmoid(numpy.dot(W2, L1_features) + b2)
#         print 'L1_features',L1_features.shape
#         print 'L2_features',L2_features.shape
#         print 'output_layer',output_layer.shape
        ground_truth = self.getGroundTruth(output)
        """ Compute the class probabilities for each example """
        theta_x       = numpy.dot(theta3, L2_features)
        hypothesis    = numpy.exp(theta_x)      
        probabilities = hypothesis / numpy.sum(hypothesis, axis = 0)
#         print 'probabilities',probabilities.shape
        diff = ground_truth - probabilities
        
        sum_of_squares_error = 0.5 * numpy.sum(numpy.multiply(diff, diff)) / input.shape[1]
        weight_decay         = 0.5 * self.lamda * (numpy.sum(numpy.multiply(W1, W1)) +
                                                   numpy.sum(numpy.multiply(W2, W2)) + 
                                                   numpy.sum(numpy.multiply(theta3, theta3)))
        cost                 = sum_of_squares_error + weight_decay 
        
        delta_out = -numpy.dot(numpy.dot(theta3.T, diff), \
                               numpy.dot((1 - probabilities).T,probabilities))
#         print 'delta_out',delta_out.shape
        delta_hid2 = numpy.dot(numpy.dot(W2.T, delta_out),\
                               numpy.dot((1 - L2_features).T, L2_features))
        delta_hid1 = numpy.dot(numpy.dot(W1.T, delta_hid2),\
                               numpy.dot((1 - L1_features).T, L1_features))
#         print 'delta_hid2',delta_hid2.shape
        theta3_grad = -numpy.dot(diff, L2_features.T)
        W2_grad = numpy.dot(delta_out, L1_features.T)
        W1_grad = numpy.dot(delta_hid2, input.T)
        b2_grad = numpy.sum(delta_out, axis = 1)
        b1_grad = numpy.sum(delta_hid2, axis = 1)
        
#         print 'W2_grad',W2_grad.shape
#         print 'W1_grad',W1_grad.shape
#         print 'b1_grad',b1_grad.shape
#         print 'b2_grad',b2_grad.shape
        
        
        
        theta3_grad = theta3_grad / input.shape[1] + self.lamda * theta3
        W2_grad = W2_grad / input.shape[1] + self.lamda * W2
        W1_grad = W1_grad / input.shape[1] + self.lamda * W1
        b1_grad = b1_grad / input.shape[1]
        b2_grad = b2_grad / input.shape[1]
        
        W1_grad = numpy.array(W1_grad)
        W2_grad = numpy.array(W2_grad)
        b1_grad = numpy.array(b1_grad)
        b2_grad = numpy.array(b2_grad)
        theta3_grad = numpy.array(theta3_grad)
        
        theta_grad = numpy.concatenate((W1_grad.flatten(), W2_grad.flatten(),
                                        b1_grad.flatten(), b2_grad.flatten(),theta3_grad.flatten()))
        print 'cost',cost
        print 'theta_grad',numpy.sum(theta_grad)
        return [cost, theta_grad]
    
    
def loadMNISTImages(file_name):
    """ Loads the images from the provided file name """
    """ Open the file """

    image_file = open(file_name, 'rb')
    
    """ Read header information from the file """
    
    head1 = image_file.read(4)
    head2 = image_file.read(4)
    head3 = image_file.read(4)
    head4 = image_file.read(4)
    

    """ Format the header information for useful data """
    
    num_examples = struct.unpack('>I', head2)[0]
    num_rows     = struct.unpack('>I', head3)[0]
    num_cols     = struct.unpack('>I', head4)[0]

    """ Initialize dataset as array of zeros """
    if num_examples > 1500:
        num_examples = 1500
        
    dataset = numpy.zeros((num_rows*num_cols, num_examples))
    
    """ Read the actual image data """
    
    images_raw  = array.array('B', image_file.read())
    image_file.close()
    
    """ Arrange the data in columns """
    
    for i in range(num_examples):
        limit1 = num_rows * num_cols * i
        limit2 = num_rows * num_cols * (i + 1)         
        dataset[:, i] = images_raw[limit1 : limit2]
    """ Normalize and return the dataset """    
    
    return dataset / 255


    
def loadMNISTLabels(file_name):
    """ Loads the image labels from the provided file name """

    """ Open the file """

    label_file = open(file_name, 'rb')
    
    """ Read header information from the file """
    
    head1 = label_file.read(4)
    head2 = label_file.read(4)
    
    """ Format the header information for useful data """
    
    num_examples = struct.unpack('>I', head2)[0]
    
    """ Initialize data labels as array of zeros """
    
    labels = numpy.zeros((num_examples, 1), dtype = numpy.int)
    
    """ Read the label data """
    
    labels_raw = array.array('b', label_file.read())
    label_file.close()
    
    """ Copy and return the label data """
    
    labels[:, 0] = labels_raw[:]
    
    if num_examples>1500:
        num_examples=1500
        
    labels = labels[0:num_examples]
    return labels

if __name__ == '__main__':
    images = loadMNISTImages('train-images.idx3-ubyte')
    labels = loadMNISTLabels('train-labels.idx1-ubyte')
    print '#1# Load train images set:',images.shape[1],'\n'
    print '#2# Load train labels set:',labels.shape[0],'\n'
    input_size = 28 * 28
    num_labels = 10
    hidden_size_1 = 196
    hidden_size_2 = 196
    sparsity_param = 0.1  # desired average activation of the hidden units.
    lambda_ = 0.0001  # weight decay parameter
    beta = 3  # weight of sparsity penalty term
    max_iterations_1 = 10
    max_iterations_2 = 10
    max_iterations_3 = 10
    max_iterations_4 = 1
    encoder_1 = SparseAutoencoder(input_size, hidden_size_1, sparsity_param, lambda_, beta)
    opt_solution  = scipy.optimize.minimize(encoder_1.sparseAutoencoderCost, encoder_1.theta, 
                                            args = (images,), method = 'L-BFGS-B', 
                                            jac = True, options = {'maxiter': max_iterations_1})
    opt_theta1 = opt_solution.x
    print '#3# Train L1 layer weight ',opt_theta1.shape,'\n'
    L1_features = encoder_1.calc_hiddenlayer(opt_theta1,images)
    print '#4# Extract Level1 features ',L1_features.shape,'\n'
    encoder_2 = SparseAutoencoder(hidden_size_1, hidden_size_2, sparsity_param, lambda_, beta)
    opt_solution  = scipy.optimize.minimize(encoder_2.sparseAutoencoderCost, encoder_2.theta, 
                                            args = (L1_features,), method = 'L-BFGS-B', 
                                            jac = True, options = {'maxiter': max_iterations_2})
    opt_theta2 = opt_solution.x
    print '#5# Train L2 layer weight ',opt_theta2.shape,'\n'
    L2_features = encoder_2.calc_hiddenlayer(opt_theta2,L1_features)
    print '#6# Extract Level2 features ',L2_features.shape,'\n'
    regressor = SoftmaxRegression(hidden_size_2, num_labels, lambda_)
     
    opt_solution  = scipy.optimize.minimize(regressor.softmaxCost, regressor.theta, 
                                            args = (L2_features, labels), method = 'L-BFGS-B', 
                                            jac = True, options = {'maxiter': max_iterations_3})
    opt_theta3     = opt_solution.x
    print '#7# Train L3 layer weight ',opt_theta3.shape,'\n'
    test_images = loadMNISTImages('t10k-images.idx3-ubyte')
    test_labels = loadMNISTLabels('t10k-labels.idx1-ubyte')
    
    print '#8# Load test images set:',test_images.shape[1],'\n'
    print '#9# Load test labels set:',test_labels.shape[0],'\n'
    
    L1_features = encoder_1.calc_hiddenlayer(opt_theta1, test_images)
    L2_features = encoder_2.calc_hiddenlayer(opt_theta2, L1_features)
    b_predictions = regressor.softmaxPredict(opt_theta3, L2_features)
     
    """ Print accuracy of the trained model """
     
    b_correct = test_labels[:, 0] == b_predictions[:, 0]
    print '#10# Before fine-tuneing accuracy :',numpy.mean(b_correct)*100,'\n'
    
    
    """Fine-tuning"""
     
    bp = BackPropagation(input_size, hidden_size_1, hidden_size_2, num_labels, \
                         opt_theta1, opt_theta2, opt_theta3, lambda_)
    opt_solution  = scipy.optimize.minimize(bp.backPropagationCost, bp.theta, 
                                            args = (images, labels), method = 'L-BFGS-B', 
                                            jac = True, options = {'maxiter': max_iterations_4})
    opt_theta4     = opt_solution.x
     
    a_predictions = bp.bpPredict(opt_theta4, test_images)
     
    a_correct = test_labels[:, 0] == a_predictions[:, 0]
    print '#11# After fine-tuneing accuracy :', numpy.mean(a_correct)*100
