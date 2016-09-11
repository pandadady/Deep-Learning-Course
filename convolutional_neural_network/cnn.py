# -*- coding:  utf-8 -*
"""
This module is used to ....
Authors: shiduo
Date:2016年9月9日
"""
import scipy.io
import scipy.signal
import scipy.sparse
import scipy.optimize
import pickle
import numpy
import sys
import time
import datetime

def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))

def test_cnn_convolve(hidden_size, visible_size, 
                      W,b, train_images,
                      patch_dim, zca_white,
                      patch_mean, image_dim):
    """
    # Checking your convolution, compare the results of your convolution with
    #  activations from the sparse autoencoder.
    :param hidden_size: hidden node number of sparse autoencoder
    :param visible_size: input node number of sparse autoencoder, the number of RGB,8*8*3
    :param train_images: large images to convolve with, matrix in the form images(r, c, channel, image number)
    :param opt_theta: theta of the sparse autoencoder
    :param image_dim: image dimension
    :param zca_white: zca whitening
    :param patch_mean: mean of the images
    :return:
    """
   
    conv_images = train_images[:, :, :, 0:8]
 
    convolved_features = cnn_convolve(patch_dim, hidden_size, conv_images,
                                          W, b, zca_white, patch_mean)

    # For 1000 random points
    for i in range(1000):
        feature_num = numpy.random.randint(0, hidden_size)
        image_num = numpy.random.randint(0, 8)
        image_row = numpy.random.randint(0, image_dim - patch_dim + 1)
        image_col = numpy.random.randint(0, image_dim - patch_dim + 1)
     
        patch = conv_images[image_row:image_row + patch_dim, image_col:image_col + patch_dim, :, image_num]
     
        patch = numpy.concatenate((patch[:, :, 0].flatten(), patch[:, :, 1].flatten(), patch[:, :, 2].flatten()))
        patch = numpy.reshape(patch, (patch.size, 1))
        patch = patch - numpy.tile(patch_mean, (patch.shape[1], 1)).transpose()
        patch = zca_white.dot(patch)
        # Number of training examples
        m = patch.shape[1]
        # Forward propagation
        z2 = W.dot(patch) + numpy.tile(b, (m, 1)).transpose()
#         print numpy.tile(b1, (m, 1)).transpose().shape
#         print b1.shape
        features = sigmoid(z2)
     
        if abs(features[feature_num, 0] - convolved_features[feature_num, image_num, image_row, image_col]) > 1e-9:
            print 'Convolved feature does not match activation from autoencoder'
            print 'Feature Number      :', feature_num
            print 'Image Number        :', image_num
            print 'Image Row           :', image_row
            print 'Image Column        :', image_col
 
            sys.exit("Convolved feature does not match activation from autoencoder. Exiting...")
    print 'Congratulations! Your convolution code passed the test.'

def cnn_convolve(patch_dim, num_features, images, W, b, zca_white, patch_mean):
    """
    Returns the convolution of the features given by W and b with
    the given images
    :param patch_dim: patch (feature) dimension
    :param num_features: number of features
    :param images: large images to convolve with, matrix in the form
                   images(r, c, channel, image number)
    :param W: weights of the sparse autoencoder
    :param b: bias of the sparse autoencoder
    :param zca_white: zca whitening
    :param patch_mean: mean of the images
    :return:
    """
#     print 'zca_white',zca_white.shape
#     print 'patch_mean',patch_mean.shape
#     print 'W',W.shape
#     print 'b',b.shape
#     print 'images',images.shape
#     print 'num_features',num_features 
#     print 'patch_dim',patch_dim 
    num_images = images.shape[3]
    image_dim = images.shape[0]
    image_channels = images.shape[2]

    #  Instructions:
    #    Convolve every feature with every large image here to produce the
    #    numFeatures x numImages x (imageDim - patchDim + 1) x (imageDim - patchDim + 1)
    #    matrix convolvedFeatures, such that
    #    convolvedFeatures(featureNum, imageNum, imageRow, imageCol) is the
    #    value of the convolved featureNum feature for the imageNum image over
    #    the region (imageRow, imageCol) to (imageRow + patchDim - 1, imageCol + patchDim - 1)
    #
    #  Expected running times:
    #    Convolving with 100 images should take less than 3 minutes
    #    Convolving with 5000 images should take around an hour
    #    (So to save time when testing, you should convolve with less images, as
    #    described earlier)

    convolved_features = numpy.zeros(shape=(num_features, num_images, image_dim - patch_dim + 1,
                                         image_dim - patch_dim + 1),
                                  dtype=numpy.float64)
#     print 'convolved_features',convolved_features.shape
    WT = W.dot(zca_white)
    bT = b - WT.dot(patch_mean)
#     print 'WT',WT.shape
    for i in range(num_images):
        for j in range(num_features):
            # convolution of image with feature matrix for each channel
            convolved_image = numpy.zeros(shape=(image_dim - patch_dim + 1, image_dim - patch_dim + 1),
                                       dtype=numpy.float64)

            for channel in range(image_channels):
                # Obtain the feature (patchDim x patchDim) needed during the convolution
                patch_size = patch_dim * patch_dim
                feature = WT[j, patch_size * channel:patch_size * (channel + 1)].reshape(patch_dim, patch_dim)
#                 print 'j',j,'patch_size * channel',patch_size * channel,'patch_size * (channel + 1)',patch_size * (channel + 1)
                # Flip the feature matrix because of the definition of convolution, as explained later
                feature = numpy.flipud(numpy.fliplr(feature))

                # Obtain the image
                im = images[:, :, channel, i]

                # Convolve "feature" with "im", adding the result to convolvedImage
                # be sure to do a 'valid' convolution
                convolved_image += scipy.signal.convolve2d(im, feature, mode='valid')
            # Subtract the bias unit (correcting for the mean subtraction as well)
            # Then, apply the sigmoid function to get the hidden activation
            convolved_image = sigmoid(convolved_image + bT[j])

            # The convolved feature is the sum of the convolved values for all channels
            convolved_features[j, i, :, :] = convolved_image

    return convolved_features

def test_cnn_pool():
    ## STEP 2c: Implement pooling
    #  Implement pooling in the function cnnPool in cnnPool.m
    
    # NOTE: Implement cnnPool in cnnPool.m first!
    
    ## STEP 2d: Checking your pooling
    #  To ensure that you have implemented pooling, we will use your pooling
    #  function to pool over a test matrix and check the results.
    test_matrix = numpy.arange(64).reshape(8, 8)
    expected_matrix = numpy.array([[numpy.mean(test_matrix[0:4, 0:4]), numpy.mean(test_matrix[0:4, 4:8])],
                                [numpy.mean(test_matrix[4:8, 0:4]), numpy.mean(test_matrix[4:8, 4:8])]])
    test_matrix = numpy.reshape(test_matrix, (1, 1, 8, 8))
    
    pooled_features = cnn_pool(4, test_matrix)
    
    if not (pooled_features == expected_matrix).all():
        print "Pooling incorrect"
        print "Expected matrix"
        print expected_matrix
        print "Got"
        print pooled_features
    
    print 'Congratulations! Your pooling code passed the test.'
    
def cnn_pool(pool_dim, convolved_features):
    """
    Pools the given convolved features

    :param pool_dim: dimension of the pooling region
    :param convolved_features: convolved features to pool (as given by cnn_convolve)
                               convolved_features(feature_num, image_num, image_row, image_col)
    :return: pooled_features: matrix of pooled features in the form
                              pooledFeatures(featureNum, imageNum, poolRow, poolCol)
    """

    num_images = convolved_features.shape[1]
    num_features = convolved_features.shape[0]
    convolved_dim = convolved_features.shape[2]

    assert convolved_dim % pool_dim == 0, "Pooling dimension is not an exact multiple of convolved dimension"

    pool_size = convolved_dim / pool_dim
    pooled_features = numpy.zeros(shape=(num_features, num_images, pool_size, pool_size),
                               dtype=numpy.float64)

    for i in range(pool_size):
        for j in range(pool_size):
            pool = convolved_features[:, :, i * pool_dim:(i + 1) * pool_dim, j * pool_dim:(j + 1) * pool_dim]
            pooled_features[:, :, i, j] = numpy.mean(numpy.mean(pool, 2), 2)

    return pooled_features
def conovlve_pool(W,b,hidden_size,image_dim,patch_dim,pool_dim,zca_white, 
                  patch_mean,num_train_images,num_test_images,train_images,test_images):
    step_size = 200
    assert hidden_size % step_size == 0, "step_size should divide hidden_size"
    
    
    
    pooled_features_train = numpy.zeros(shape=(hidden_size, num_train_images,
                                            numpy.floor((image_dim - patch_dim + 1) / pool_dim),
                                            numpy.floor((image_dim - patch_dim + 1) / pool_dim)),
                                     dtype=numpy.float64)
    pooled_features_test = numpy.zeros(shape=(hidden_size, num_test_images,
                                           numpy.floor((image_dim - patch_dim + 1) / pool_dim),
                                           numpy.floor((image_dim - patch_dim + 1) / pool_dim)),
                                    dtype=numpy.float64)
    
    start_time = time.time()
    for conv_part in range(hidden_size / step_size):
        features_start = conv_part * step_size
        features_end = (conv_part + 1) * step_size
        print "\tround", conv_part, "features", features_start, "to", features_end
    
        Wt = W[features_start:features_end, :]
        bt = b[features_start:features_end]
    
        print "\tconvolving & pooling train images"
        convolved_features = cnn_convolve(patch_dim, step_size, train_images,
                                              Wt, bt, zca_white, patch_mean)
        pooled_features = cnn_pool(pool_dim, convolved_features)
        pooled_features_train[features_start:features_end, :, :, :] = pooled_features
    
        print "\ttime elapsed:", str(datetime.timedelta(seconds=time.time() - start_time))
    
        print "\tconvolving and pooling test images"
        convolved_features = cnn_convolve(patch_dim, step_size, test_images,
                                              Wt, bt, zca_white, patch_mean)
        pooled_features = cnn_pool(pool_dim, convolved_features)
        pooled_features_test[features_start:features_end, :, :, :] = pooled_features
    
        print "\ttime elapsed:", str(datetime.timedelta(seconds=time.time() - start_time))
    
    
    print "\tSaved",'pooled_features_train',pooled_features_train.shape,'pooled_features_test',pooled_features_test.shape
    print "\ttime elapsed:", str(datetime.timedelta(seconds=time.time() - start_time))
    return  pooled_features_train,pooled_features_test
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
#         print ground_truth.shape
#         print probabilities.shape
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


def executeCnn():
    rho            = 0.035   # desired average activation of hidden units
    lamda          = 3e-3    # weight decay parameter
    image_channels = 3
    image_dim = 64  # image dimension
    patch_dim = 8
    visible_size = patch_dim * patch_dim * image_channels  # number of input units, used to parse
    hidden_size = 400 # used to parse
    max_iterations = 1000
   
    pool_dim = 19  # dimension of pooling region
    print "Step 1: loading linear_decoder's theta, zca whiten features and mean of patch, which are learned from stlSampledPatches.mat "
    with open('stl10_features.pickle', 'r') as f:
        opt_theta = pickle.load(f)
        zca_white = pickle.load(f)
        patch_mean = pickle.load(f)
    print "\topt_theta is ",opt_theta.shape
    print "\tzca_white is ",zca_white.shape
    print "\tpatch_mean is ",patch_mean.shape
    W = opt_theta[0:hidden_size * visible_size].reshape(hidden_size, visible_size)
    b = opt_theta[2 * hidden_size * visible_size:2 * hidden_size * visible_size + hidden_size]
    print "\tW is ",W.shape
    print "\tb is ",b.shape
      
    print "Step 2: loading stlTrainSubset.mat and stlTestSubset.mat"
    stl_train = scipy.io.loadmat('data/stlTrainSubset.mat')
    train_images = stl_train['trainImages']
    train_images = train_images[:,:,:,:50]
    print "\ttrain_images shape is ",train_images.shape
    train_labels = stl_train['trainLabels']
    train_labels = train_labels[:50,:]
    print "\ttrain_labels shape is ",train_labels.shape
    num_train_images = stl_train['numTrainImages'][0][0]
    num_train_images = 50
    print "\tnum_train_images is ",num_train_images
    stl_test = scipy.io.loadmat('data/stlTestSubset.mat')
    test_images = stl_test['testImages']
    test_images = test_images[:,:,:,:50]
    print "\ttest_images shape is ",test_images.shape
    test_labels = stl_test['testLabels']
    test_labels = test_labels[:50,:]
    print "\ttest_labels shape is ",test_labels.shape
    num_test_images = stl_test['numTestImages'][0][0]
    num_test_images = 50
    print "\tnum_test_images is ",num_test_images
    print "Step 3: test convolve "
    test_cnn_convolve(hidden_size, visible_size, 
                          W,b, train_images,
                          patch_dim, zca_white,
                          patch_mean, image_dim)
    print "Step 4: test pool "
    test_cnn_pool()
    print "Step 5: conovlve and pool "
    pooled_features_train,pooled_features_test = conovlve_pool(W,b,hidden_size,image_dim,patch_dim,pool_dim,zca_white, 
                  patch_mean,num_train_images,num_test_images,train_images,test_images)
    print('\tSaving pooled features...')
    with open('cnn_pooled_features.pickle', 'wb') as f:
        pickle.dump(pooled_features_train, f)
        pickle.dump(pooled_features_test, f)
        pickle.dump(train_labels, f)
        pickle.dump(test_labels, f)
#           
    print "Step 6: Softmax"
    # Load pooled features
    with open('cnn_pooled_features.pickle', 'r') as f:
        pooled_features_train = pickle.load(f)
        pooled_features_test = pickle.load(f)
        train_labels = pickle.load(f)
        test_labels = pickle.load(f)  
    # Setup parameters for softmax
    softmax_lambda = 1e-4
    num_classes = 4
    num_train_images = 50
    max_iterations = 1000
    # Reshape the pooled_features to form an input vector for softmax
    softmax_train_images =numpy.transpose(pooled_features_train, axes=[0, 2, 3, 1])
    softmax_train_images = softmax_train_images.reshape((softmax_train_images.size / num_train_images, num_train_images))
    softmax_train_labels = train_labels.flatten() - 1  # Ensure that labels are from 0..n-1 (for n classes)
    """ Initialize Softmax Regressor with the above parameters """
    
    regressor = SoftmaxRegression(softmax_train_images.size / num_train_images, num_classes, softmax_lambda)
    
    """ Run the L-BFGS algorithm to get the optimal parameter values """
    
    opt_solution  = scipy.optimize.minimize(regressor.softmaxCost, regressor.theta, 
                                            args = (softmax_train_images, softmax_train_labels), method = 'L-BFGS-B', 
                                            jac = True, options = {'maxiter': max_iterations})
    opt_theta     = opt_solution.x
    softmax_test_images =numpy.transpose(pooled_features_train, axes=[0, 2, 3, 1])
    softmax_test_images = softmax_test_images.reshape((softmax_test_images.size / num_train_images, num_train_images))
    softmax_test_labels = test_labels.flatten() - 1  # Ensure that labels are from 0..n-1 (for n classes)
    print     softmax_test_labels.shape 

    predictions = regressor.softmaxPredict(opt_theta, softmax_test_images)
    softmax_test = numpy.zeros((softmax_test_labels.shape[0], 1))
    softmax_test[:, 0] = numpy.argmax(softmax_test_labels, axis = 0)
    print predictions
    print softmax_test
    correct = softmax_test[:, 0] == predictions[:, 0]
    print """Accuracy :""", numpy.mean(correct)
if __name__ == '__main__':
    executeCnn()
