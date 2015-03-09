# -*- coding: utf-8 -*-
"""
Created on Wed Nov 06 01:29:36 2013

@author: Michael
"""

from numpy import *
import scipy.io
from scipy import optimize
from scipy.stats import bernoulli
import gnumpy as gpu


def mlpSoftmax_costfunc(x, *args):
    numClasses, inputSize, l1Size, l2Size, l3Size, lambda_softmax, lambda_hidden, inputs, labels, groundTruth, dropout_probability = args
    numCases = shape(inputs)[1]
    num_weights_L1 = l1Size * (inputSize + 1)
    num_weights_L2 = l2Size * (l1Size + 1)
    num_weights_L3 = l3Size * (l2Size + 1)
    num_weights_softmax = numClasses * l3Size
    #x = gpu.garray(x)
    inputs = gpu.garray(inputs)
    theta_L1 = gpu.garray(reshape(x[0:num_weights_L1], (l1Size, inputSize + 1)))
    #theta_L1 = x[0:num_weights_L1].reshape((l1Size, inputSize + 1))
    #print numClasses, l2Size
    theta_L2 = gpu.garray(reshape(x[num_weights_L1:num_weights_L2+num_weights_L1], (l2Size, l1Size + 1)))
    #theta_L2 = x[num_weights_L1:num_weights_L2+num_weights_L1].reshape((l2Size, l1Size + 1))
    theta_L3 = gpu.garray(reshape(x[num_weights_L2+num_weights_L1:num_weights_L2+num_weights_L1+num_weights_L3], (l3Size, l2Size + 1)))
    theta_softmax = gpu.garray(reshape(x[num_weights_L2+num_weights_L1+num_weights_L3:shape(x)[0]], (numClasses, l3Size)))
    #theta_softmax = x[num_weights_L2+num_weights_L1:shape(x)[0]].reshape((numClasses, l2Size))
    theta_L1_grad = gpu.zeros(shape(theta_L1))
    theta_L2_grad = gpu.zeros(shape(theta_L2))
    theta_L3_grad = gpu.zeros(shape(theta_L3))
    dropout_l1 = gpu.garray(bernoulli.rvs(dropout_probability, size = (l1Size+1, numCases)))
    dropout_l2 = gpu.garray(bernoulli.rvs(dropout_probability, size = (l2Size+1, numCases)))
    dropout_l3 = gpu.garray(bernoulli.rvs(dropout_probability, size = (l3Size, numCases)))
    inputs = gpu.concatenate((gpu.ones((1,numCases)), inputs), axis = 0)
    hidden_sum_L1 = gpu.dot(theta_L1, inputs)
    #hidden_activation_L1 = gpu.log(1+hidden_sum_L1.exp())
    relu_mask_hidden1 = gpu.ones(shape(hidden_sum_L1)) * (hidden_sum_L1>0)
    hidden_activation_L1 = hidden_sum_L1*relu_mask_hidden1
    hidden_derivative_L1 = relu_mask_hidden1
    #hidden_activation_L1 = gpu.concatenate((gpu.ones((1,numCases)), hidden_activation_L1), axis=0)
    hidden_derivative_L1 = gpu.concatenate((gpu.ones((1,numCases)), hidden_derivative_L1), axis=0)
    hidden_activation_L1 = gpu.concatenate((gpu.ones((1,numCases)), hidden_activation_L1), axis=0) * dropout_l1
    hidden_sum_L2 = gpu.dot(theta_L2, hidden_activation_L1)
    #hidden_activation_L2 = gpu.log(1+hidden_sum_L2.exp())
    relu_mask_hidden2 = gpu.ones(shape(hidden_sum_L2)) * (hidden_sum_L2>0)
    hidden_activation_L2 = hidden_sum_L2*relu_mask_hidden2
    hidden_derivative_L2 = relu_mask_hidden2
    #hidden_activation_L2 = gpu.concatenate((gpu.ones((1,numCases)), hidden_activation_L2), axis=0)
    hidden_derivative_L2 = gpu.concatenate((gpu.ones((1,numCases)), hidden_derivative_L2), axis=0)
    hidden_activation_L2 = gpu.concatenate((gpu.ones((1,numCases)), hidden_activation_L2), axis=0) * dropout_l2
    hidden_sum_L3 = gpu.dot(theta_L3, hidden_activation_L2)
    #hidden_activation_L3 = gpu.log(1+hidden_sum_L3.exp())
    relu_mask_hidden3 = gpu.ones(shape(hidden_sum_L3)) * (hidden_sum_L3>0)
    #hidden_activation_L3 = hidden_sum_L3*relu_mask_hidden3
    hidden_derivative_L3 = relu_mask_hidden3
    hidden_activation_L3 = hidden_sum_L3*relu_mask_hidden3 * dropout_l3
    #hidden_activation_L3 = hidden_sum_L3.logistic() * dropout_l3
    hidden_sum_softmax = gpu.dot(theta_softmax, hidden_activation_L3)
    hidden_sum_softmax = hidden_sum_softmax - hidden_sum_softmax.max(axis = 0)
    predictions = hidden_sum_softmax.exp()
    predictions = predictions / gpu.sum(predictions,axis = 0)
    pred = predictions.argmax(axis=0) + 1
    accuracy = mean(pred == labels) * 100
    temp = groundTruth*gpu.log(predictions)
    temp = temp.as_numpy_array()
    temp[temp==-inf] = -200.0
    temp = nan_to_num(temp)
    regularized_penalty_L1 = theta_L1[:,1:shape(theta_L1)[1]]
    regularized_penalty_L2 = theta_L2[:,1:shape(theta_L2)[1]]
    regularized_penalty_L3 = theta_L3[:,1:shape(theta_L3)[1]]
    regularized_penalty_L1 = regularized_penalty_L1 * regularized_penalty_L1
    regularized_penalty_L2 = regularized_penalty_L2 * regularized_penalty_L2
    regularized_penalty_L3 = regularized_penalty_L3 * regularized_penalty_L3
    pred_cost = -1*sum(temp)/numCases
    l2norm_cost = 0.5 * lambda_hidden*(gpu.sum(regularized_penalty_L3) + gpu.sum(regularized_penalty_L2) + gpu.sum(regularized_penalty_L1)) + 0.5 * lambda_softmax * gpu.sum(theta_softmax*theta_softmax)
    #l2norm_cost = 0
    cost = pred_cost + l2norm_cost
    print 'Prediction Accuracy:                       ', accuracy, '%'
    print 'Multilayer Softmax Prediction Cost:        ', pred_cost
    print 'Multilayer Softmax L2 Normalisation Cost:  ', l2norm_cost
    print 'Multilayer Softmax Cost:                   ', cost    
    print '--------------------------------------------------------------------'
    softmax_imd = groundTruth - predictions
    #theta_softmax_grad = -1*gpu.dot(softmax_imd, gpu.garray(transpose(hidden_activation_L3.as_numpy_array())))/numCases
    theta_softmax_grad = -1*gpu.dot(softmax_imd, gpu.garray(transpose(hidden_activation_L3.as_numpy_array())))/numCases + lambda_softmax * theta_softmax
    deltaOut = -softmax_imd
    delta_L3_imd = gpu.dot(gpu.garray(transpose(theta_softmax.as_numpy_array())), deltaOut)
    delta_L3_imd2 = delta_L3_imd*hidden_derivative_L3
    #delta_L3_imd2 = (delta_L3_imd * hidden_activation_L3) * (1-hidden_activation_L3)
    delta_L3 = gpu.dot(delta_L3_imd2, gpu.garray(transpose(hidden_activation_L2.as_numpy_array())))
    theta_L3_grad += delta_L3
    delta_L2_imd = gpu.dot(gpu.garray(transpose(theta_L3.as_numpy_array())), delta_L3_imd2)
    delta_L2_imd2 = delta_L2_imd*hidden_derivative_L2
    delta_L2_imd2 = delta_L2_imd2[1:shape(delta_L2_imd2)[0]+1, :]
    delta_L2 = gpu.dot(delta_L2_imd2, gpu.garray(transpose(hidden_activation_L1.as_numpy_array())))
    theta_L2_grad += delta_L2
    delta_L1_imd = gpu.dot(gpu.garray(transpose(theta_L2.as_numpy_array())), delta_L2_imd2)
    delta_L1_imd2 = delta_L1_imd*hidden_derivative_L1
    delta_L1_imd2 = delta_L1_imd2[1:shape(delta_L1_imd2)[0]+1, :]
    delta_L1 = gpu.dot(delta_L1_imd2, gpu.garray(transpose(inputs.as_numpy_array())))
    theta_L1_grad += delta_L1
    theta_L1_grad = theta_L1_grad/numCases
    theta_L2_grad = theta_L2_grad/numCases
    theta_L3_grad = theta_L3_grad/numCases
    theta_L1_grad[:, 1:shape(theta_L1_grad)[1]] = theta_L1_grad[:, 1:shape(theta_L1_grad)[1]] + theta_L1[:, 1: shape(theta_L1)[1]] * lambda_hidden
    theta_L2_grad[:, 1:shape(theta_L2_grad)[1]] = theta_L2_grad[:, 1:shape(theta_L2_grad)[1]] + theta_L2[:, 1: shape(theta_L2)[1]] * lambda_hidden
    theta_L3_grad[:, 1:shape(theta_L3_grad)[1]] = theta_L3_grad[:, 1:shape(theta_L3_grad)[1]] + theta_L3[:, 1: shape(theta_L3)[1]] * lambda_hidden       
    theta_L1_grad = reshape(theta_L1_grad.as_numpy_array(), num_weights_L1)
    theta_L2_grad = reshape(theta_L2_grad.as_numpy_array(), num_weights_L2)
    theta_L3_grad = reshape(theta_L3_grad.as_numpy_array(), num_weights_L3)
    theta_softmax_grad = reshape(theta_softmax_grad.as_numpy_array(), num_weights_softmax)
    del inputs
    del theta_L1
    del theta_L2
    del theta_L3
    del theta_softmax
    del hidden_sum_L1
    del hidden_activation_L1
    del hidden_sum_L2
    del hidden_activation_L2
    del hidden_activation_L3
    del hidden_sum_L3
    del hidden_sum_softmax
    del predictions
    del temp
    del softmax_imd
    del deltaOut
    del delta_L3_imd
    del delta_L3_imd2
    del delta_L3
    del delta_L2_imd
    del delta_L2_imd2
    del delta_L2
    del delta_L1_imd
    del delta_L1_imd2
    del delta_L1
    #del regularized_penalty_L1
    #del regularized_penalty_L2
    gpu.free_reuse_cache()
    return cost, hstack((theta_L1_grad,theta_L2_grad,theta_L3_grad,theta_softmax_grad))
    
#def mlpSoftmax_grad(x, *args):
#    numClasses, inputSize, l1Size, l2Size, l3Size, lambda_softmax,lambda_hidden, inputs, labels, groundTruth,drop_l1,drop_l2,drop_l3 = args
#    numCases = shape(inputs)[1]
#    num_weights_L1 = l1Size * (inputSize + 1)
#    num_weights_L2 = l2Size * (l1Size + 1)
#    num_weights_L3 = l3Size * (l2Size + 1)
#    num_weights_softmax = numClasses * l3Size
#    inputs = gpu.garray(inputs)
#    theta_L1 = gpu.garray(reshape(x[0:num_weights_L1], (l1Size, inputSize + 1)))
#    theta_L2 = gpu.garray(reshape(x[num_weights_L1:num_weights_L2+num_weights_L1], (l2Size, l1Size + 1)))
#    theta_L3 = gpu.garray(reshape(x[num_weights_L2+num_weights_L1:num_weights_L2+num_weights_L1+num_weights_L3], (l3Size, l2Size + 1)))
#    theta_softmax = gpu.garray(reshape(x[num_weights_L2+num_weights_L1+num_weights_L3:shape(x)[0]], (numClasses, l3Size)))
#    theta_L1_grad = gpu.zeros(shape(theta_L1))
#    theta_L2_grad = gpu.zeros(shape(theta_L2))
#    theta_L3_grad = gpu.zeros(shape(theta_L3))
#    inputs = gpu.concatenate((gpu.ones((1,numCases)), inputs), axis = 0)
#    hidden_sum_L1 = gpu.dot(theta_L1, inputs)
#    #hidden_activation_L1 = gpu.log(1+hidden_sum_L1.exp())
#    #hidden_derivative_L1 = hidden_sum_L1.logistic()    
#    relu_mask_hidden1 = gpu.ones(shape(hidden_sum_L1)) * (hidden_sum_L1>0)
#    hidden_activation_L1 = hidden_sum_L1*relu_mask_hidden1    
#    hidden_derivative_L1 = relu_mask_hidden1
#    hidden_activation_L1 = gpu.concatenate((gpu.ones((1,numCases)), hidden_activation_L1), axis=0)
#    #hidden_activation_L1 = gpu.concatenate((gpu.ones((1,numCases)), hidden_activation_L1), axis=0) * drop_l1
#    hidden_derivative_L1 = gpu.concatenate((gpu.ones((1,numCases)), hidden_derivative_L1), axis=0)
#    hidden_sum_L2 = gpu.dot(theta_L2, hidden_activation_L1)
#    #hidden_activation_L2 = gpu.log(1+hidden_sum_L2.exp())
#    relu_mask_hidden2 = gpu.ones(shape(hidden_sum_L2)) * (hidden_sum_L2>0)
#    hidden_activation_L2 = hidden_sum_L2*relu_mask_hidden2
#    #hidden_derivative_L2 = hidden_sum_L2.logistic()
#    hidden_derivative_L2 = relu_mask_hidden2
#    hidden_activation_L2 = gpu.concatenate((gpu.ones((1,numCases)), hidden_activation_L2), axis=0)
#    #hidden_activation_L2 = gpu.concatenate((gpu.ones((1,numCases)), hidden_activation_L2), axis=0) * drop_l2
#    hidden_derivative_L2 = gpu.concatenate((gpu.ones((1,numCases)), hidden_derivative_L2), axis=0)
#    hidden_sum_L3 = gpu.dot(theta_L3, hidden_activation_L2)
#    #hidden_activation_L3 = gpu.log(1+hidden_sum_L3.exp())
#    relu_mask_hidden3 = gpu.ones(shape(hidden_sum_L3)) * (hidden_sum_L3>0)
#    hidden_activation_L3 = hidden_sum_L3*relu_mask_hidden3
#    #hidden_activation_L3 = hidden_sum_L3*relu_mask_hidden3 * drop_l3
#    #hidden_derivative_L3 = hidden_sum_L3.logistic()
#    hidden_derivative_L3 = relu_mask_hidden3
#    hidden_sum_softmax_imd = gpu.dot(theta_softmax, hidden_activation_L3)
#    #hidden_softmax_activation = hidden_sum_softmax_imd.logistic()
#    hidden_sum_softmax = hidden_sum_softmax_imd - hidden_sum_softmax_imd.max(axis = 0)
#    predictions = hidden_sum_softmax.exp()
#    predictions = predictions / gpu.sum(predictions,axis = 0)
#    softmax_imd = groundTruth - predictions
#    theta_softmax_grad = -1*gpu.dot(softmax_imd, gpu.garray(transpose(hidden_activation_L3.as_numpy_array())))/numCases + lambda_softmax * theta_softmax
#    deltaOut = -softmax_imd
#    delta_L3_imd = gpu.dot(gpu.garray(transpose(theta_softmax.as_numpy_array())), deltaOut)
#    delta_L3_imd2 = delta_L3_imd*hidden_derivative_L3
#    delta_L3 = gpu.dot(delta_L3_imd2, gpu.garray(transpose(hidden_activation_L2.as_numpy_array())))
#    theta_L3_grad += delta_L3
#    delta_L2_imd = gpu.dot(gpu.garray(transpose(theta_L3.as_numpy_array())), delta_L3_imd2)
#    delta_L2_imd2 = delta_L2_imd*hidden_derivative_L2
#    delta_L2 = gpu.dot(delta_L2_imd2, gpu.garray(transpose(hidden_activation_L1.as_numpy_array())))
#    theta_L2_grad += delta_L2[1:shape(theta_L2)[0]+1,:]
#    delta_L1_imd = gpu.dot(gpu.garray(transpose(theta_L2.as_numpy_array())), delta_L2_imd2[1:shape(delta_L2_imd2)[0], :])
#    delta_L1_imd2 = delta_L1_imd*hidden_derivative_L1
#    delta_L1 = gpu.dot(delta_L1_imd2, gpu.garray(transpose(inputs.as_numpy_array())))
#    theta_L1_grad += delta_L1[1:shape(theta_L1)[0]+1,:]
#    theta_L1_grad = theta_L1_grad/numCases
#    theta_L2_grad = theta_L2_grad/numCases
#    theta_L3_grad = theta_L3_grad/numCases
#    #theta_L1_grad[:, 1:shape(theta_L1_grad)[1]] = theta_L1_grad[:, 1:shape(theta_L1_grad)[1]] + theta_L1[:, 1: shape(theta_L1)[1]] * lambda_hidden
#    #theta_L2_grad[:, 1:shape(theta_L2_grad)[1]] = theta_L2_grad[:, 1:shape(theta_L2_grad)[1]] + theta_L2[:, 1: shape(theta_L2)[1]] * lambda_hidden
#    theta_L3_grad[:, 1:shape(theta_L3_grad)[1]] = theta_L3_grad[:, 1:shape(theta_L3_grad)[1]] + theta_L3[:, 1: shape(theta_L3)[1]] * lambda_hidden       
#    theta_L1_grad = reshape(theta_L1_grad.as_numpy_array(), num_weights_L1)
#    theta_L2_grad = reshape(theta_L2_grad.as_numpy_array(), num_weights_L2)
#    theta_L3_grad = reshape(theta_L3_grad.as_numpy_array(), num_weights_L3)
#    theta_softmax_grad = reshape(theta_softmax_grad.as_numpy_array(), num_weights_softmax)
#    del inputs
#    del theta_L1
#    del theta_L2
#    del theta_L3
#    del theta_softmax
#    del hidden_sum_L1
#    del hidden_activation_L1
#    del hidden_sum_L2
#    del hidden_activation_L2
#    del hidden_activation_L3
#    del hidden_sum_L3
#    del hidden_sum_softmax
#    del predictions
#    del softmax_imd
#    del deltaOut
#    del delta_L3_imd
#    del delta_L3_imd2
#    del delta_L3
#    del delta_L2_imd
#    del delta_L2_imd2
#    del delta_L2
#    del delta_L1_imd
#    del delta_L1_imd2
#    del delta_L1
#    gpu.free_reuse_cache()
#    return hstack((theta_L1_grad,theta_L2_grad,theta_L3_grad,theta_softmax_grad))

def ground_Truth(labels,numCases):
    temp_truth = zeros((labels.max(), numCases))
    for n in range(size(labels)):
        temp_truth[labels[n][0]-1][n] = 1
    return gpu.garray(temp_truth)

def obtain_data():
    inpu = scipy.io.loadmat('input.mat')
    inputs = inpu['inputData']
    label = scipy.io.loadmat('labels.mat')
    labels = label['labels']
    trainData = inputs[:,0:45000]
    trainLabels = labels[0:45000]
    testData = inputs[:,45000:shape(inputs)[1]]
    testLabels = labels[45000:shape(labels)[0]]
    return trainData, trainLabels, testData, testLabels

def mlpSoftmax_test():
    numClasses = 10
    inputSize = 28 * 28
    l1Size = 100
    l2Size = 30
    l3Size = 20
    lambda_softmax = 7e-4
    lambda_hidden = 2e-6
    print "Loading data..."
    inputs, labels, testData, testLabels = obtain_data()
    print "Done."
    numCases = shape(inputs)[1]
    num_weights_L1 = l1Size * (inputSize + 1)
    num_weights_L2 = l2Size * (l1Size + 1)
    num_weights_L3 = l3Size * (l2Size + 1)
    num_weights_softmax = numClasses * l3Size
    r = sqrt(6)/sqrt(inputSize+l1Size+l2Size+l3Size+1)
    theta_L1 = (gpu.rand(l1Size, inputSize+1))*2*r-r
    theta_L2 = (gpu.rand(l2Size, l1Size+1))*2*r-r
    theta_L3 = (gpu.rand(l3Size, l2Size+1))*2*r-r
    theta_softmax = (gpu.rand(numClasses, l3Size))*2*r-r
    groundTruth = ground_Truth(labels,numCases)
    #theta_L1 = reshape(theta_L1, num_weights_L1)
    theta_L1 = theta_L1.reshape(num_weights_L1)
    #theta_L2 = reshape(theta_L2, num_weights_L2)
    theta_L2 = theta_L2.reshape(num_weights_L2)
    #theta_softmax = reshape(theta_softmax, num_weights_softmax)
    theta_L3 = theta_L3.reshape(num_weights_L3)
    theta_softmax = theta_softmax.reshape(num_weights_softmax)
    theta = hstack((theta_L1.as_numpy_array(), theta_L2.as_numpy_array(), theta_L3.as_numpy_array(), theta_softmax.as_numpy_array()))
    dropout_probability = 1
    args = (numClasses, inputSize, l1Size, l2Size, l3Size, lambda_softmax, lambda_hidden, inputs, squeeze(labels), groundTruth,dropout_probability)
    print "Starting Network Training..."
    opttheta = optimize.fmin_l_bfgs_b(mlpSoftmax_costfunc, theta, args=args, maxiter=350)
    theta = opttheta[0]
    print "Training finished."
    scipy.io.savemat('mlpSoftmax.mat', mdict={'theta': theta})
    print "Now testing prediction accuracy..."
    theta_L1 = reshape(theta[0:num_weights_L1], (l1Size, inputSize + 1))
    theta_L2 = reshape(theta[num_weights_L1:num_weights_L2+num_weights_L1], (l2Size, l1Size + 1))
    theta_L3 = reshape(theta[num_weights_L2+num_weights_L1:num_weights_L2+num_weights_L1+num_weights_L3], (l3Size, l2Size + 1))
    theta_softmax = reshape(theta[num_weights_L2+num_weights_L1+num_weights_L3:shape(theta)[0]], (numClasses, l3Size))
    numCasesPred = shape(testData)[1]
    testData = concatenate((ones((1,numCasesPred)), testData), axis = 0)
    hidden_sum_L1 = dot(theta_L1, testData)
    #hidden_activation_L1 = log(1+exp(hidden_sum_L1))
    relu_mask_hidden1 = ones(shape(hidden_sum_L1)) * (hidden_sum_L1>0)
    hidden_activation_L1 = hidden_sum_L1*relu_mask_hidden1
    hidden_activation_L1 = concatenate((ones((1,numCasesPred)), hidden_activation_L1), axis=0)
    hidden_sum_L2 = dot(theta_L2, hidden_activation_L1)
    #hidden_activation_L2 = log(1+exp(hidden_sum_L2))
    relu_mask_hidden2 = ones(shape(hidden_sum_L2)) * (hidden_sum_L2>0)
    hidden_activation_L2 = hidden_sum_L2*relu_mask_hidden2 
    hidden_activation_L2 = concatenate((ones((1,numCasesPred)), hidden_activation_L2), axis=0)
    hidden_sum_L3 = dot(theta_L3, hidden_activation_L2)
    #hidden_activation_L3 = log(1+exp(hidden_sum_L3))
    relu_mask_hidden3 = ones(shape(hidden_sum_L3)) * (hidden_sum_L3>0)
    hidden_activation_L3 = hidden_sum_L3*relu_mask_hidden3  
    hidden_sum_softmax = dot(theta_softmax, hidden_activation_L3)
    hidden_sum_softmax = hidden_sum_softmax - hidden_sum_softmax.max(axis = 0)
    predictions = exp(hidden_sum_softmax)
    predictions = predictions / predictions.sum(axis = 0)
    pred = predictions.argmax(axis=0) + 1
    testLabels = squeeze(testLabels)
    accuracy = mean(pred == testLabels) * 100
    print "Accuracy: ", accuracy, "%"
    return pred, testLabels
    
def checkGradient():
    inputSize = 64
    l1Size = 5
    l2Size = 5
    l3Size = 4
    numCases = 5
    numClasses = 3
    lambda_softmax = 1e-4
    lambda_hidden = 1e-5
    inputs = random.rand(inputSize, 5)
    labels = array([[1],[2],[3],[2],[1]])
    r = sqrt(6)/sqrt(inputSize+l1Size+l2Size+l3Size+1)
    theta_L1 = (random.rand(l1Size, inputSize+1))*2*r-r
    theta_L2 = (random.rand(l2Size, l1Size+1))*2*r-r
    theta_L3 = (random.rand(l3Size, l2Size+1))*2*r-r
    theta_softmax = (random.rand(numClasses, l3Size))*2*r-r
    num_weights_L1 = l1Size * (inputSize + 1)
    num_weights_L2 = l2Size * (l1Size + 1)
    num_weights_L3 = l3Size * (l2Size + 1)
    num_weights_softmax = numClasses * l3Size
    theta_L1 = reshape(theta_L1, num_weights_L1)
    theta_L2 = reshape(theta_L2, num_weights_L2)
    theta_L3 = reshape(theta_L3, num_weights_L3)
    theta_softmax = reshape(theta_softmax, num_weights_softmax)
    theta = hstack((theta_L1, theta_L2, theta_L3, theta_softmax))
    groundTruth = ground_Truth(labels,numCases)
    dropout_probability = 1
    args = (numClasses, inputSize, l1Size, l2Size,l3Size,lambda_softmax, lambda_hidden, inputs, labels, groundTruth,dropout_probability)
    numgrad = zeros(size(theta))
    perturb = zeros(size(theta))
    e = 1e-4
    for p in range(size(theta)):
        perturb[p] = e;
        minus_weights = theta - perturb
        plus_weights = theta + perturb
        loss1 = mlpSoftmax_costfunc(minus_weights, *args)
        loss2 = mlpSoftmax_costfunc(plus_weights, *args)
        numgrad[p] = (loss2 - loss1) / (2*e)
        perturb[p] = 0
    grad = mlpSoftmax_grad(theta, *args)
    diff = linalg.norm(numgrad-grad)/linalg.norm(numgrad+grad)
    return diff

#mlpSoftmax_test()
