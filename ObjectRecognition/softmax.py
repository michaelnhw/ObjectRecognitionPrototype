# -*- coding: utf-8 -*-
"""
Created on Wed Nov 06 18:41:42 2013

@author: Michael
"""
from numpy import *
import scipy.io
from scipy import optimize
import time

def softmax_cost(x, *args):
    numClasses, inputSize, lambda_val, data, groundTruth = args
    theta = reshape(x, (numClasses, inputSize))
    numCases = shape(data)[1]
    weight_sum = dot(theta, data)
    weight_sum = weight_sum - weight_sum.max(axis=0)
    predictions = exp(weight_sum)
    predictions = predictions / predictions.sum(axis=0)
    temp = multiply(groundTruth, log(predictions))
    cost = -1*sum(temp)/numCases + lambda_val*sum(theta**2)/2
    print 'Softmax Cost: ', cost
    return cost
    
def softmax_grad(x, *args):
    numClasses, inputSize, lambda_val, data, groundTruth = args
    theta = reshape(x, (numClasses, inputSize))
    numCases = shape(data)[1]
    weight_sum = dot(theta, data)
    weight_sum = weight_sum - weight_sum.max(axis=0)
    predictions = exp(weight_sum)
    predictions = predictions / predictions.sum(axis=0)
    delta = groundTruth-predictions
    theta_grad = -1*dot(delta, transpose(data))/numCases + lambda_val * theta
    theta_grad = reshape(theta_grad, size(x))
    return theta_grad

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
    return inputs, labels
    
def softmax_train():
    numClasses = 10
    inputSize = 28 * 28
    lambda_val = 1e-4
    data, labels = obtain_data()
    numCases = shape(data)[1]
    theta = 0.005 * random.rand(numClasses * inputSize, 1)
    groundTruth = ground_Truth(labels,numCases)
    args = (numClasses, inputSize, lambda_val, data, groundTruth)
    opttheta = optimize.fmin_l_bfgs_b(softmax_cost, theta, fprime=softmax_grad, args=args, maxiter=150, factr = 10.0)
    theta = opttheta[0]
    theta = reshape(theta, (numClasses, inputSize))
    scipy.io.savemat('softmax.mat', mdict={'theta': theta})
    return theta

def benchmark():
    start = time.clock()
    softmax_train()
    end = time.clock()
    print "execution time: ",end - start,"seconds"