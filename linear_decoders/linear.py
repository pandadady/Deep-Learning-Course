# -*- coding:  utf-8 -*
"""
This module is used to ....
Authors: shiduo
Date:2016年8月17日
"""
import math
import numpy
import time
import PIL
import scipy.io
import scipy.optimize
class LinearDecoder(object):
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

    
        
    def linearDecoderCost(self, theta, input):
        """ Returns the cost of the Autoencoder and gradient at a particular 'theta' """
        """ Extract weights and biases from 'theta' input """
        W1 = theta[self.limit0 : self.limit1].reshape(self.hidden_size, self.visible_size)
        W2 = theta[self.limit1 : self.limit2].reshape(self.visible_size, self.hidden_size)
        b1 = theta[self.limit2 : self.limit3].reshape(self.hidden_size, 1)
        b2 = theta[self.limit3 : self.limit4].reshape(self.visible_size, 1)
        
        """ Compute output layers by performing a feedforward pass
            Computation is done for all the training inputs simultaneously """
        hidden_layer = self.sigmoid(numpy.dot(W1, input) + b1)
        output_layer =  numpy.dot(W2, hidden_layer) + b2
        
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
        
        del_out = diff
        del_hid = numpy.multiply(numpy.dot(numpy.transpose(W2), del_out) + numpy.transpose(numpy.matrix(KL_div_grad)), 
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
    
def display_color_network(A, filename='weights.png'):
    """
    # display receptive field(s) or basis vector(s) for image patches
    #
    # A         the basis, with patches as column vectors

    # In case the midpoint is not set at 0, we shift it dynamically

    :param A:
    :param file:
    :return:
    """
    if numpy.min(A) >= 0:
        A = A - numpy.mean(A)

    cols = numpy.round(numpy.sqrt(A.shape[1]))

    channel_size = A.shape[0] / 3
    dim = numpy.sqrt(channel_size)
    dimp = dim + 1
    rows = numpy.ceil(A.shape[1] / cols)

    B = A[0:channel_size, :]
    C = A[channel_size:2 * channel_size, :]
    D = A[2 * channel_size:3 * channel_size, :]

    B = B / numpy.max(numpy.abs(B))
    C = C / numpy.max(numpy.abs(C))
    D = D / numpy.max(numpy.abs(D))

    # Initialization of the image
    image = numpy.ones(shape=(dim * rows + rows - 1, dim * cols + cols - 1, 3))

    for i in range(int(rows)):
        for j in range(int(cols)):
            # This sets the patch
            image[i * dimp:i * dimp + dim, j * dimp:j * dimp + dim, 0] = B[:, i * cols + j].reshape(dim, dim)
            image[i * dimp:i * dimp + dim, j * dimp:j * dimp + dim, 1] = C[:, i * cols + j].reshape(dim, dim)
            image[i * dimp:i * dimp + dim, j * dimp:j * dimp + dim, 2] = D[:, i * cols + j].reshape(dim, dim)

    image = (image + 1) / 2

    PIL.Image.fromarray(numpy.uint8(image * 255), 'RGB').save(filename)

    return 0

# this function accepts a 2D vector as input.
# Its outputs are:
#   value: h(x1, x2) = x1^2 + 3*x1*x2
#   grad: A 2x1 vector that gives the partial derivatives of h with respect to x1 and x2
# Note that when we pass @simpleQuadraticFunction(x) to computeNumericalGradients, we're assuming
# that computeNumericalGradients will use only the first returned value of this function.
def simple_quadratic_function(x):
    value = x[0] ** 2 + 3 * x[0] * x[1]

    grad = numpy.zeros(shape=2, dtype=numpy.float32)
    grad[0] = 2 * x[0] + 3 * x[1]
    grad[1] = 3 * x[0]

    return value, grad


def compute_gradient(J, theta):
    epsilon = 0.0001

    gradient = numpy.zeros(theta.shape)
    theta_epsilon_plus = numpy.array(theta, dtype=numpy.float64)
    for i in range(theta.shape[0]):
        theta_epsilon_plus = numpy.array(theta, dtype=numpy.float64)
        theta_epsilon_plus[i] = theta[i] + epsilon
        theta_epsilon_minus = numpy.array(theta, dtype=numpy.float64)
        theta_epsilon_minus[i] = theta[i] - epsilon

        gradient[i] = (J(theta_epsilon_plus)[0] - J(theta_epsilon_minus)[0]) / (2 * epsilon)
#         if i % 100 == 0:
#             print "Computing gradient for input:", i

    return gradient


def executeLinearDecoder():
    rho            = 0.01   # desired average activation of hidden units
    lamda          = 0.0001 # weight decay parameter
    beta           = 3      # weight of sparsity penalty term
    visible_size = 8  # number of input units
    hidden_size  = 5  # number of hidden units
    
    print "Step 0 : check gradient "
    training_data = numpy.random.rand(8, 10)
    encoder = LinearDecoder(visible_size, hidden_size, rho, lamda, beta)
    cost, theta_grad = encoder.linearDecoderCost(encoder.theta, training_data)
    J = lambda x: encoder.linearDecoderCost(x, training_data)
    num_grad = compute_gradient(J, encoder.theta)
    diff = numpy.linalg.norm(num_grad - theta_grad) / numpy.linalg.norm(num_grad + theta_grad)
    print '    The difference between theta_grad and num_grad is less than 1e-9'
    print '    diff:', diff, diff< 1e-9
    print "Step 1 : load training data"
    patches = scipy.io.loadmat('data/stlSampledPatches.mat')['patches']
    print "    data shape is ",patches.shape
    display_color_network(patches[:, 0:10000], filename='patches_raw.png')
    print "    save image in current path and name patches_raw.png"
    patch_mean = numpy.mean(patches, 1)
    patches = patches - numpy.tile(patch_mean, (patches.shape[1], 1)).transpose()
    print "Step 2 : ZCA whitening"
    epsilon = 0.1  # epsilon for ZCA whitening
    sigma = patches.dot(patches.transpose()) / patches.shape[1]
    (u, s, v) = numpy.linalg.svd(sigma)
    zca_white = u.dot(numpy.diag(1 / (s + epsilon))).dot(u.transpose())
    patches_zca = zca_white.dot(patches)
    display_color_network(patches_zca[:, 0:100], filename='patches_zca.png')
    print "    save image in current path and name patches_zca.png"
    print "Step 3 : training linear decoder"
    rho            = 0.035   # desired average activation of hidden units
    lamda          = 3e-3 # weight decay parameter
    beta           = 5      # weight of sparsity penalty term
    
    image_channels = 3
    patch_dim = 8
    visible_size = patch_dim * patch_dim * image_channels  # number of input units
    hidden_size  = 5  # number of hidden units
    hidden_size = 400
    max_iterations = 400
    encoder = LinearDecoder(visible_size, hidden_size, rho, lamda, beta)
    opt_solution  = scipy.optimize.minimize(encoder.linearDecoderCost, encoder.theta, 
                                            args = (patches_zca[:, 0:10000],), method = 'L-BFGS-B', 
                                            jac = True, options = {'maxiter': max_iterations})
    opt_theta     = opt_solution.x
    ## STEP 2d: Visualize learned features
    W = opt_theta[0:hidden_size * visible_size].reshape(hidden_size, visible_size)
    b = opt_theta[2 * hidden_size * visible_size:2 * hidden_size * visible_size + hidden_size]
    print "    save image in current path and name patches_zca_features.png"
    display_color_network(W.dot(zca_white).transpose(), 'patches_zca_features.png')
if __name__ == '__main__':
    executeLinearDecoder()
