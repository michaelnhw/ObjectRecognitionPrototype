# -*- coding: utf-8 -*-
"""
Created on Wed Nov 06 01:29:36 2013

@author: Michael
"""

from numpy import *
import scipy.io
from scipy import optimize
import time

def mlpSoftmax_costfunc(x, *args):
    numClasses, inputSize, l1Size, l2Size, lambda_softmax, lambda_hidden, inputs, labels, groundTruth = args
    numCases = shape(inputs)[1]
    num_weights_L1 = l1Size * (inputSize + 1)
    num_weights_L2 = l2Size * (l1Size + 1)
    theta_L1 = reshape(x[0:num_weights_L1], (l1Size, inputSize + 1))
    theta_L2 = reshape(x[num_weights_L1:num_weights_L2+num_weights_L1], (l2Size, l1Size + 1))
    theta_softmax = reshape(x[num_weights_L2+num_weights_L1:shape(x)[0]], (numClasses, l2Size))
    inputs = concatenate((ones((1,numCases)), inputs), axis = 0)
    hidden_sum_L1 = dot(theta_L1, inputs)
    hidden_activation_L1 = 1/(1 + exp(-hidden_sum_L1))
    hidden_activation_L1 = concatenate((ones((1,numCases)), hidden_activation_L1), axis=0)
    hidden_sum_L2 = dot(theta_L2, hidden_activation_L1)
    hidden_activation_L2 = 1/(1 + exp(-hidden_sum_L2))
    hidden_sum_softmax = dot(theta_softmax, hidden_activation_L2)
    hidden_sum_softmax = hidden_sum_softmax - hidden_sum_softmax.max(axis = 0)
    predictions = exp(hidden_sum_softmax)
    predictions = predictions / predictions.sum(axis = 0)
    temp = multiply(groundTruth,log(predictions))
    regularized_penalty_L1 = theta_L1[:,1:shape(theta_L1)[1]]
    regularized_penalty_L2 = theta_L2[:,1:shape(theta_L2)[1]]
    regularized_penalty_L1 = regularized_penalty_L1 ** 2
    regularized_penalty_L2 = regularized_penalty_L2 **2
    cost = -1*sum(temp)/numCases + 0.5 * lambda_hidden*(sum(regularized_penalty_L1) + sum(regularized_penalty_L2)) + 0.5 * lambda_softmax * sum(theta_softmax**2)
    print 'Multilayer Softmax Cost:', cost
    return cost
    
def mlpSoftmax_grad(x, *args):
    numClasses, inputSize, l1Size, l2Size, lambda_softmax,lambda_hidden, inputs, labels, groundTruth = args
    numCases = shape(inputs)[1]
    num_weights_L1 = l1Size * (inputSize + 1)
    num_weights_L2 = l2Size * (l1Size + 1)
    num_weights_softmax = numClasses * l2Size
    theta_L1 = reshape(x[0:num_weights_L1], (l1Size, inputSize + 1))
    theta_L2 = reshape(x[num_weights_L1:num_weights_L2+num_weights_L1], (l2Size, l1Size + 1))
    theta_softmax = reshape(x[num_weights_L2+num_weights_L1:shape(x)[0]], (numClasses, l2Size))
    theta_L1_grad = zeros(shape(theta_L1))
    theta_L2_grad = zeros(shape(theta_L2))
    inputs = concatenate((ones((1,numCases)), inputs), axis = 0)
    hidden_sum_L1 = dot(theta_L1, inputs)
    hidden_activation_L1 = 1/(1 + exp(-hidden_sum_L1))
    hidden_activation_L1 = concatenate((ones((1,numCases)), hidden_activation_L1), axis=0)
    hidden_sum_L2 = dot(theta_L2, hidden_activation_L1)
    hidden_activation_L2 = 1/(1 + exp(-hidden_sum_L2))
    hidden_sum_softmax_imd = dot(theta_softmax, hidden_activation_L2)
    hidden_softmax_activation = 1/(1 + exp(-hidden_sum_softmax_imd))
    hidden_sum_softmax = hidden_sum_softmax_imd - hidden_sum_softmax_imd.max(axis = 0)
    predictions = exp(hidden_sum_softmax)
    predictions = predictions / predictions.sum(axis = 0)
    softmax_imd = groundTruth - predictions
    theta_softmax_grad = -1*dot(softmax_imd, transpose(hidden_activation_L2))/numCases + lambda_softmax * theta_softmax
    deltaOut = -softmax_imd
    delta_L2_imd = dot(transpose(theta_softmax), deltaOut)
    delta_L2_imd2 = multiply(multiply(delta_L2_imd, hidden_activation_L2), (1-hidden_activation_L2))
    delta_L2 = dot(delta_L2_imd2, transpose(hidden_activation_L1))
    theta_L2_grad += delta_L2
    delta_L1_imd = dot(transpose(theta_L2), delta_L2_imd2)
    delta_L1_imd2 = multiply(multiply(delta_L1_imd, hidden_activation_L1), (1-hidden_activation_L1))
    delta_L1 = dot(delta_L1_imd2, transpose(inputs))
    theta_L1_grad += delta_L1[1:shape(theta_L1)[0]+1,:]
    theta_L1_grad = theta_L1_grad/numCases
    theta_L2_grad = theta_L2_grad/numCases
    theta_L1_grad[:, 1:shape(theta_L1_grad)[1]] = theta_L1_grad[:, 1:shape(theta_L1_grad)[1]] + theta_L1[:, 1: shape(theta_L1)[1]] * lambda_hidden
    theta_L2_grad[:, 1:shape(theta_L2_grad)[1]] = theta_L2_grad[:, 1:shape(theta_L2_grad)[1]] + theta_L2[:, 1: shape(theta_L2)[1]] * lambda_hidden
    theta_L1_grad = reshape(theta_L1_grad, num_weights_L1)
    theta_L2_grad = reshape(theta_L2_grad, num_weights_L2)
    theta_softmax_grad = reshape(theta_softmax_grad, num_weights_softmax)
    return hstack((theta_L1_grad,theta_L2_grad, theta_softmax_grad))

def ground_Truth(labels,numCases):
    temp_truth = zeros((labels.max(), numCases))
    for n in range(size(labels)):
        temp_truth[labels[n][0]-1][n] = 1
    return temp_truth

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
    l2Size = 20
    lambda_softmax = 1e-4
    lambda_hidden = 8e-5
    print "Loading data..."
    inputs, labels, testData, testLabels = obtain_data()
    print shape(labels)
    print "Done."
    numCases = shape(inputs)[1]
    num_weights_L1 = l1Size * (inputSize + 1)
    num_weights_L2 = l2Size * (l1Size + 1)
    num_weights_softmax = numClasses * l2Size
    r = sqrt(6)/sqrt(inputSize+l1Size+l2Size+1)
    theta_L1 = (random.rand(l1Size, inputSize+1))*2*r-r
    theta_L2 = (random.rand(l2Size, l1Size+1))*2*r-r
    theta_softmax = (random.rand(numClasses, l2Size))*2*r-r
    groundTruth = ground_Truth(labels,numCases)
    theta_L1 = reshape(theta_L1, num_weights_L1)
    theta_L2 = reshape(theta_L2, num_weights_L2)
    theta_softmax = reshape(theta_softmax, num_weights_softmax)
    theta = hstack((theta_L1, theta_L2, theta_softmax))
    args = (numClasses, inputSize, l1Size, l2Size, lambda_softmax, lambda_hidden, inputs, labels, groundTruth)
    print "Starting Network Training..."
    opttheta = optimize.fmin_l_bfgs_b(mlpSoftmax_costfunc, theta, fprime=mlpSoftmax_grad, args=args, maxiter=400)
    theta = opttheta[0]
    print "Training finished."
    scipy.io.savemat('mlpSoftmax.mat', mdict={'theta': theta})
    print "Now testing prediction accuracy..."
    theta_L1 = reshape(theta[0:num_weights_L1], (l1Size, inputSize + 1))
    theta_L2 = reshape(theta[num_weights_L1:num_weights_L2+num_weights_L1], (l2Size, l1Size + 1))
    theta_softmax = reshape(theta[num_weights_L2+num_weights_L1:shape(theta)[0]], (numClasses, l2Size))
    numCasesPred = shape(testData)[1]
    testData = concatenate((ones((1,numCasesPred)), testData), axis = 0)
    hidden_sum_L1 = dot(theta_L1, testData)
    hidden_activation_L1 = 1/(1 + exp(-hidden_sum_L1))
    hidden_activation_L1 = concatenate((ones((1,numCasesPred)), hidden_activation_L1), axis=0)
    hidden_sum_L2 = dot(theta_L2, hidden_activation_L1)
    hidden_activation_L2 = 1/(1 + exp(-hidden_sum_L2))
    hidden_sum_softmax = dot(theta_softmax, hidden_activation_L2)
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
    numCases = 5
    numClasses = 3
    lambda_softmax = 1e-4
    lambda_hidden = 1e-5
    inputs = random.rand(inputSize, 5)
    labels = array([[1],[2],[3],[2],[1]])
    r = sqrt(6)/sqrt(inputSize+l1Size+l2Size+1)
    theta_L1 = (random.rand(l1Size, inputSize+1))*2*r-r
    theta_L2 = (random.rand(l2Size, l1Size+1))*2*r-r
    theta_softmax = (random.rand(numClasses, l2Size))*2*r-r
    num_weights_L1 = l1Size * (inputSize + 1)
    num_weights_L2 = l2Size * (l1Size + 1)
    num_weights_softmax = numClasses * l2Size
    theta_L1 = reshape(theta_L1, num_weights_L1)
    theta_L2 = reshape(theta_L2, num_weights_L2)
    theta_softmax = reshape(theta_softmax, num_weights_softmax)
    theta = hstack((theta_L1, theta_L2, theta_softmax))
    groundTruth = ground_Truth(labels,numCases)
    args = (numClasses, inputSize, l1Size, l2Size, lambda_softmax, lambda_hidden, inputs, labels, groundTruth)
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
