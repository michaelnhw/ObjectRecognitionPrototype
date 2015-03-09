# -*- coding: utf-8 -*-
"""
Created on Wed Nov 06 01:29:36 2013

@author: Michael
"""

from numpy import *
import scipy.io
from scipy import optimize
import time
import gnumpy as gpu

def multi_costfunc(x, *args):
    num_input,num_hidden1, num_hidden2, num_output,inputs,targets,lambda_val = args
    num_weights1 = (num_input+1)*num_hidden1
    num_weights2 = (num_hidden1+1)*num_hidden2
    weights1 = reshape(x[0:num_weights1],(num_hidden1,num_input+1))
    weights2 = reshape(x[num_weights1:num_weights1+num_weights2], (num_hidden2,num_hidden1+1))
    weights3 = reshape(x[num_weights1+num_weights2:shape(x)[0]], (num_output,num_hidden2+1))
    nData = shape(inputs)[1]
    x = concatenate((ones((1,nData)), inputs), axis = 0)
    hidden1_sum = dot(weights1, x)
    hidden1_activation = 1/(1 + exp(-hidden1_sum))
    hidden1_activation = concatenate((ones((1,nData)), hidden1_activation), axis = 0)
    hidden2_sum = dot(weights2, hidden1_activation)
    hidden2_activation = 1/(1 + exp(-hidden2_sum))
    hidden2_activation = concatenate((ones((1,nData)), hidden2_activation), axis = 0)
    output_sum = dot(weights3, hidden2_activation)
    outputs = 1/(1 + exp(-output_sum))
    regularized_penalty1 = weights1[:,1:shape(weights1)[1]]
    regularized_penalty2 = weights2[:,1:shape(weights2)[1]]
    regularized_penalty3 = weights3[:,1:shape(weights3)[1]]
    regularized_penalty1 = regularized_penalty1 ** 2
    regularized_penalty2 = regularized_penalty2 ** 2
    regularized_penalty3 = regularized_penalty3 ** 2
    output_target_diff = (outputs - targets)**2
    cost = sum(output_target_diff)/(2*nData) + 0.5 * lambda_val * (sum(regularized_penalty1) + sum(regularized_penalty2) + sum(regularized_penalty3))
    print 'Multilayer Cost: ', cost   
    return cost
    
def multi_grad_costfunc(x, *args):
    num_input,num_hidden1, num_hidden2, num_output,inputs,targets,lambda_val = args
    num_weights1 = (num_input+1)*num_hidden1
    num_weights2 = (num_hidden1+1)*num_hidden2
    num_weights3 = (num_hidden2+1)*num_output
    weights1 = reshape(x[0:num_weights1],(num_hidden1,num_input+1))
    weights2 = reshape(x[num_weights1:num_weights1+num_weights2], (num_hidden2,num_hidden1+1))
    weights3 = reshape(x[num_weights1+num_weights2:shape(x)[0]], (num_output,num_hidden2+1))
    nData = shape(inputs)[1]
    x = concatenate((ones((1,nData)), inputs), axis = 0)
    hidden1_sum = dot(weights1, x)
    hidden1_activation = 1/(1 + exp(-hidden1_sum))
    hidden1_activation = concatenate((ones((1,nData)), hidden1_activation), axis = 0)
    hidden2_sum = dot(weights2, hidden1_activation)
    hidden2_activation = 1/(1 + exp(-hidden2_sum))
    hidden2_activation = concatenate((ones((1,nData)), hidden2_activation), axis = 0)
    output_sum = dot(weights3, hidden2_activation)
    outputs = 1/(1 + exp(-output_sum))
    weights1_grad = zeros(shape(weights1))
    weights2_grad = zeros(shape(weights2))
    weights3_grad = zeros(shape(weights3))
    p = multiply(multiply((outputs - targets), outputs), (1-outputs))
    weights3_grad += dot(p, transpose(hidden2_activation))
    q_temp = dot(transpose(weights3),p)
    q = multiply(multiply(q_temp,hidden2_activation),(1-hidden2_activation))
    delta2 = dot(q, transpose(hidden1_activation))
    weights2_grad += delta2[1:shape(delta2)[0], :]
    e_temp = dot(transpose(weights2), q[1:shape(q)[0], :])
    e = multiply(multiply(e_temp,hidden1_activation),(1-hidden1_activation))
    delta3 = dot(e, transpose(x))
    weights1_grad += delta3[1:shape(delta3)[0], :]
    weights1_grad = weights1_grad/nData
    weights2_grad = weights2_grad/nData
    weights3_grad = weights3_grad/nData
    weights1_grad[:,1:shape(weights1_grad)[1]] = weights1_grad[:,1:shape(weights1_grad)[1]] + weights1[:,1:shape(weights1)[1]] * lambda_val
    weights2_grad[:,1:shape(weights2_grad)[1]] = weights2_grad[:,1:shape(weights2_grad)[1]] + weights2[:,1:shape(weights2)[1]] * lambda_val
    weights3_grad[:,1:shape(weights3_grad)[1]] = weights3_grad[:,1:shape(weights3_grad)[1]] + weights3[:,1:shape(weights3)[1]] * lambda_val
    weights1_grad = reshape(weights1_grad, num_weights1)
    weights2_grad = reshape(weights2_grad, num_weights2)
    weights3_grad = reshape(weights3_grad, num_weights3)
    return hstack((weights1_grad,weights2_grad,weights3_grad))
    
def mlpSingleOutput1Layer_costfunc(x, *args):
    inputSize, l1Size, lambda_hidden, inputs, targets = args
    numCases = shape(inputs)[1]
    num_weights_L1 = l1Size * (inputSize + 1)
    inputs = gpu.garray(inputs)
    targets = gpu.garray(targets)
    theta_L1 = gpu.garray(reshape(x[0:num_weights_L1], (l1Size, inputSize + 1)))
    theta_output = gpu.garray(reshape(x[num_weights_L1:shape(x)[0]], (1, l1Size+1)))
    inputs = gpu.concatenate((gpu.ones((1,numCases)), inputs), axis = 0)
    hidden_sum_L1 = gpu.dot(theta_L1, inputs)
    hidden_activation_L1 = hidden_sum_L1.logistic()
    hidden_activation_L1 = gpu.concatenate((gpu.ones((1,numCases)), hidden_activation_L1), axis = 0)
    #hidden_activation_L1 = hidden_activation_L1 * dropout_prob
    hidden_sum_output = gpu.dot(theta_output, hidden_activation_L1)
    outputs = hidden_sum_output.logistic()
    output_target_diff = (outputs - targets)**2
    regularized_penalty_output = theta_output[:,1:shape(theta_output)[1]]
    regularized_penalty_output = regularized_penalty_output * regularized_penalty_output
    regularized_penalty_L1 = theta_L1[:,1:shape(theta_L1)[1]]
    regularized_penalty_L1 = regularized_penalty_L1 * regularized_penalty_L1
    cost = gpu.sum(output_target_diff)/(2*numCases) + 0.5 * lambda_hidden*(gpu.sum(regularized_penalty_L1)+gpu.sum(regularized_penalty_output))
    print 'Multilayer Preceptron Cost:', cost
    del inputs
    del theta_L1
    del hidden_sum_L1
    del hidden_activation_L1
    del regularized_penalty_output
    del regularized_penalty_L1
    gpu.free_reuse_cache()
    return cost

def mlpSingleOutput1Layer_grad(x, *args):
    inputSize, l1Size, lambda_hidden, inputs, targets = args
    numCases = shape(inputs)[1]
    num_weights_L1 = l1Size * (inputSize + 1)
    num_weights_output = 1 * (l1Size+1)
    inputs = gpu.garray(inputs)
    targets = gpu.garray(targets)
    theta_L1 = gpu.garray(reshape(x[0:num_weights_L1], (l1Size, inputSize + 1)))
    theta_output = gpu.garray(reshape(x[num_weights_L1:shape(x)[0]], (1, l1Size+1)))
    inputs = gpu.concatenate((gpu.ones((1,numCases)), inputs), axis = 0)
    hidden_sum_L1 = gpu.dot(theta_L1, inputs)
    hidden_activation_L1 = hidden_sum_L1.logistic()
    hidden_activation_L1 = gpu.concatenate((gpu.ones((1,numCases)), hidden_activation_L1), axis = 0)
    #hidden_activation_L1 = hidden_activation_L1 * dropout_prob
    hidden_sum_output = gpu.dot(theta_output, hidden_activation_L1)
    outputs = hidden_sum_output.logistic()
    theta_L1_grad = gpu.zeros(shape(theta_L1))
    theta_output_grad = gpu.zeros(shape(theta_output))
    a = (outputs - targets) * outputs * (1-outputs)
    theta_output_grad += gpu.dot(a, gpu.garray(transpose(hidden_activation_L1.as_numpy_array())))
    b_temp = gpu.dot(gpu.garray(transpose(theta_output.as_numpy_array())),a)
    b = (b_temp*hidden_activation_L1)*(1-hidden_activation_L1)
    delta2 = gpu.dot(b, gpu.garray(transpose(inputs.as_numpy_array())))
    theta_L1_grad += delta2[1:shape(delta2)[0], :]
    theta_L1_grad = theta_L1_grad/numCases
    theta_output_grad = theta_output_grad/numCases
    theta_output_grad[:,1:shape(theta_output_grad)[1]] = theta_output_grad[:,1:shape(theta_output_grad)[1]] + theta_output[:,1:shape(theta_output)[1]] * lambda_hidden
    theta_L1_grad[:,1:shape(theta_L1_grad)[1]] = theta_L1_grad[:,1:shape(theta_L1_grad)[1]] + theta_L1[:,1:shape(theta_L1)[1]] * lambda_hidden
    theta_output_grad = reshape(theta_output_grad.as_numpy_array(), num_weights_output)
    theta_L1_grad = reshape(theta_L1_grad.as_numpy_array(), num_weights_L1)
    del inputs
    del theta_L1
    del hidden_sum_L1
    del hidden_activation_L1
    gpu.free_reuse_cache()
    return hstack((theta_L1_grad,theta_output_grad))
    
#def checkGradient():
#    num_input = 64
#    num_hidden = 5
#    num_output = 64
#    lambda_val = 0.0001
#    sparsityParam = 0.01
#    beta = 3
#    img = sampleImg()
#    inputs = img[:,0:10]
#    r = sqrt(6)/sqrt(num_hidden+num_input+1)
#    weights1 = (random.rand(num_hidden,num_input+1))*2*r-r
#    weights2 = (random.rand(num_output,num_hidden+1))*2*r-r
#    num_weights1 = (num_input+1)*num_hidden
#    num_weights2 = (num_hidden+1)*num_output
#    weights1 = reshape(weights1, num_weights1)
#    weights2 = reshape(weights2, num_weights2)
#    weights = hstack((weights1,weights2))
#    args = (num_input,num_hidden,num_output,inputs,lambda_val,sparsityParam,beta)
#    numgrad = zeros(size(weights))
#    perturb = zeros(size(weights))
#    e = 1e-4
#    for p in range(size(weights)):
#        perturb[p] = e;
#        minus_weights = weights - perturb
#        plus_weights = weights + perturb
#        loss1 = costfunc(minus_weights, *args)
#        loss2 = costfunc(plus_weights, *args)
#        numgrad[p] = (loss2 - loss1) / (2*e)
#        perturb[p] = 0
#    grad = grad_costfunc(weights, *args)
#    diff = norm(numgrad-grad)/norm(numgrad+grad)
#    return diff
    
def multi_train():
    X = array([[0.3755,0.2210,0.9848,1,0,0.4572,0.7407,0.1561,0.5780,0.0430,0.6437,0.8278]])
    y = array([[0.0269,0,0.9305,1,0.0459,0.0266,0.3796,0.0404,0.0720,0.0717,0.1810,0.6051]])
    num_input = 1
    num_hidden1 = 5
    num_output = 1
    lambda_val = 0
    weights1 = (random.rand(num_hidden1, num_input+1))*2*0.12-0.12
    weights2 = (random.rand(num_output, num_hidden1+1))*2*0.12-0.12
    num_weights1 = (num_input+1)*num_hidden1
    num_weights2 = (num_hidden1+1)*num_output
    weights1 = reshape(weights1, num_weights1)
    weights2 = reshape(weights2, num_weights2)
    weights = hstack((weights1,weights2))
    args = (num_input,num_hidden1, lambda_val, X, y)
    opttheta = optimize.fmin_l_bfgs_b(mlpSingleOutput1Layer_costfunc, weights, fprime=mlpSingleOutput1Layer_grad, args=args, maxiter=100)
    weights = opttheta[0]
    weights1 = reshape(weights[0:num_weights1],(num_hidden1, num_input + 1))
    weights2 = reshape(weights[num_weights1:shape(weights)[0]], (num_output, num_hidden1+1))
    x = concatenate((ones((1,12)), X), axis = 0)
    hidden1_sum = dot(weights1, x)
    hidden1_activation = 1/(1 + exp(-hidden1_sum))
    hidden1_activation = concatenate((ones((1,12)), hidden1_activation), axis = 0)
    hidden2_sum = dot(weights2, hidden1_activation)
    hidden2_activation = 1/(1 + exp(-hidden2_sum))
    return hidden2_activation

#start = time.clock()
#result = multi_train()
#end = time.clock()
#print end - start
