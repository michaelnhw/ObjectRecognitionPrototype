# -*- coding: utf-8 -*-
"""
Created on Wed Nov 06 01:29:36 2013

@author: Michael
"""

from numpy import *
import scipy.io
from scipy import optimize
import gnumpy as gpu

def mlpSoftmax1Layer_costfunc(x, *args):
    numClasses, inputSize, l1Size, lambda_softmax, lambda_hidden, inputs, groundTruth = args
    numCases = shape(inputs)[1]
    num_weights_L1 = l1Size * (inputSize + 1)
    inputs = gpu.garray(inputs)
    theta_L1 = gpu.garray(reshape(x[0:num_weights_L1], (l1Size, inputSize + 1)))
    theta_softmax = gpu.garray(reshape(x[num_weights_L1:shape(x)[0]], (numClasses, l1Size)))
    inputs = gpu.concatenate((gpu.ones((1,numCases)), inputs), axis = 0)
    hidden_sum_L1 = gpu.dot(theta_L1, inputs)
    #hidden_activation_L1 = gpu.log(1+hidden_sum_L1.exp())
    relu_mask_hidden1 = gpu.ones(shape(hidden_sum_L1)) * (hidden_sum_L1>0)
    hidden_activation_L1 = hidden_sum_L1*relu_mask_hidden1
    #hidden_activation_L1 = hidden_sum_L1.logistic()
    hidden_sum_softmax = gpu.dot(theta_softmax, hidden_activation_L1)
    hidden_sum_softmax = hidden_sum_softmax - hidden_sum_softmax.max(axis = 0)
    predictions = hidden_sum_softmax.exp()
    predictions = predictions / gpu.sum(predictions,axis = 0)
    temp = groundTruth*gpu.log(predictions)
    temp = temp.as_numpy_array()
    temp[temp==-inf] = -200.0
    temp = nan_to_num(temp)
    regularized_penalty_L1 = theta_L1[:,1:shape(theta_L1)[1]]
    regularized_penalty_L1 = regularized_penalty_L1 * regularized_penalty_L1
    cost = -1*sum(temp)/numCases + 0.5 * lambda_hidden*(gpu.sum(regularized_penalty_L1)) + 0.5 * lambda_softmax * gpu.sum(theta_softmax*theta_softmax)
    print 'Multilayer Softmax Cost:', cost
    del inputs
    del theta_L1
    del theta_softmax
    del hidden_sum_L1
    del hidden_activation_L1
    del hidden_sum_softmax
    del predictions
    del temp
    del regularized_penalty_L1
    gpu.free_reuse_cache()
    return cost
    
def mlpSoftmax1Layer_grad(x, *args):
    numClasses, inputSize, l1Size,lambda_softmax,lambda_hidden, inputs, groundTruth = args
    numCases = shape(inputs)[1]
    num_weights_L1 = l1Size * (inputSize + 1)
    num_weights_softmax = numClasses * l1Size
    inputs = gpu.garray(inputs)
    theta_L1 = gpu.garray(reshape(x[0:num_weights_L1], (l1Size, inputSize + 1)))
    theta_softmax = gpu.garray(reshape(x[num_weights_L1:shape(x)[0]], (numClasses, l1Size)))
    theta_L1_grad = gpu.zeros(shape(theta_L1))
    inputs = gpu.concatenate((gpu.ones((1,numCases)), inputs), axis = 0)
    hidden_sum_L1 = gpu.dot(theta_L1, inputs)
    #hidden_activation_L1 = gpu.log(1+hidden_sum_L1.exp())
    #hidden_derivative_L1 = hidden_sum_L1.logistic()
    relu_mask_hidden1 = gpu.ones(shape(hidden_sum_L1)) * (hidden_sum_L1>0)
    hidden_activation_L1 = hidden_sum_L1*relu_mask_hidden1
    #hidden_activation_L1 = hidden_sum_L1.logistic()
    hidden_derivative_L1 = relu_mask_hidden1
    hidden_sum_softmax_imd = gpu.dot(theta_softmax, hidden_activation_L1)
    hidden_sum_softmax = hidden_sum_softmax_imd - hidden_sum_softmax_imd.max(axis = 0)
    predictions = hidden_sum_softmax.exp()
    predictions = predictions / gpu.sum(predictions,axis = 0)
    softmax_imd = groundTruth - predictions
    theta_softmax_grad = -1*gpu.dot(softmax_imd, gpu.garray(transpose(hidden_activation_L1.as_numpy_array())))/numCases + lambda_softmax * theta_softmax
    deltaOut = -softmax_imd
    delta_L1_imd = gpu.dot(gpu.garray(transpose(theta_softmax.as_numpy_array())), deltaOut)
    delta_L1_imd2 = delta_L1_imd*hidden_derivative_L1
    #delta_L1_imd2 = (delta_L1_imd*hidden_activation_L1)*(1-hidden_activation_L1)
    delta_L1 = gpu.dot(delta_L1_imd2, gpu.garray(transpose(inputs.as_numpy_array())))
    theta_L1_grad += delta_L1
    theta_L1_grad = theta_L1_grad/numCases
    theta_L1_grad[:, 1:shape(theta_L1_grad)[1]] = theta_L1_grad[:, 1:shape(theta_L1_grad)[1]] + theta_L1[:, 1: shape(theta_L1)[1]] * lambda_hidden
    theta_L1_grad = reshape(theta_L1_grad.as_numpy_array(), num_weights_L1)
    theta_softmax_grad = reshape(theta_softmax_grad.as_numpy_array(), num_weights_softmax)
    del inputs
    del theta_L1
    del theta_softmax
    del hidden_sum_L1
    del hidden_activation_L1
    del hidden_sum_softmax
    del predictions
    del softmax_imd
    del deltaOut
    del delta_L1_imd
    del delta_L1_imd2
    del delta_L1
    gpu.free_reuse_cache()
    return hstack((theta_L1_grad,theta_softmax_grad))

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
    lambda_softmax = 7e-4
    lambda_hidden = 8e-6
    print "Loading data..."
    inputs, labels, testData, testLabels = obtain_data()
    print shape(labels)
    print "Done."
    numCases = shape(inputs)[1]
    num_weights_L1 = l1Size * (inputSize + 1)
    num_weights_softmax = numClasses * l1Size
    r = gpu.sqrt(6)/gpu.sqrt(inputSize+l1Size+1)
    theta_L1 = (gpu.rand(l1Size, inputSize+1))*2*r-r
    theta_softmax = (gpu.rand(numClasses, l1Size))*2*r-r
    groundTruth = ground_Truth(labels,numCases)
    theta_L1 = theta_L1.reshape(num_weights_L1)
    theta_softmax = theta_softmax.reshape(num_weights_softmax)
    theta = hstack((theta_L1.as_numpy_array(), theta_softmax.as_numpy_array()))
    args = (numClasses, inputSize, l1Size,lambda_softmax, lambda_hidden, inputs, groundTruth)
    print "Starting Network Training..."
    opttheta = optimize.fmin_l_bfgs_b(mlpSoftmax1Layer_costfunc, theta, fprime=mlpSoftmax1Layer_grad, args=args, maxiter=300)
    theta = opttheta[0]
    print "Training finished."
    scipy.io.savemat('mlpSoftmax.mat', mdict={'theta': theta})
    print "Now testing prediction accuracy..."
    theta_L1 = reshape(theta[0:num_weights_L1], (l1Size, inputSize + 1))
    theta_softmax = reshape(theta[num_weights_L1:shape(theta)[0]], (numClasses, l1Size))
    numCasesPred = shape(testData)[1]
    testData = concatenate((ones((1,numCasesPred)), testData), axis = 0)
    hidden_sum_L1 = dot(theta_L1, testData)
    #hidden_activation_L1 = log(1+exp(hidden_sum_L1))
    relu_mask_hidden1 = ones(shape(hidden_sum_L1)) * (hidden_sum_L1>0)
    hidden_activation_L1 = hidden_sum_L1*relu_mask_hidden1 
    hidden_sum_softmax = dot(theta_softmax, hidden_activation_L1)
    hidden_sum_softmax = hidden_sum_softmax - hidden_sum_softmax.max(axis = 0)
    predictions = exp(hidden_sum_softmax)
    predictions = predictions / predictions.sum(axis = 0)
    pred = predictions.argmax(axis=0) + 1
    testLabels = squeeze(testLabels)
    accuracy = mean(pred == testLabels) * 100
    print "Accuracy: ", accuracy, "%"
    return accuracy
    
def checkGradient():
    inputSize = 64
    l1Size = 5
    numCases = 5
    numClasses = 3
    lambda_softmax = 1e-4
    lambda_hidden = 1e-5
    inputs = random.rand(inputSize, 5)
    labels = array([[1],[2],[3],[2],[1]])
    r = sqrt(6)/sqrt(inputSize+l1Size+1)
    theta_L1 = (random.rand(l1Size, inputSize+1))*2*r-r
    theta_softmax = (random.rand(numClasses, l1Size))*2*r-r
    num_weights_L1 = l1Size * (inputSize + 1)
    num_weights_softmax = numClasses * l1Size
    theta_L1 = reshape(theta_L1, num_weights_L1)
    theta_softmax = reshape(theta_softmax, num_weights_softmax)
    theta = hstack((theta_L1, theta_softmax))
    groundTruth = ground_Truth(labels,numCases)
    args = (numClasses, inputSize, l1Size,lambda_softmax, lambda_hidden, inputs, labels, groundTruth)
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
    
