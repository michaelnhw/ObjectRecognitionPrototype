# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 21:21:01 2014

@author: himwaing
"""

from numpy import *
from scipy import optimize
# from autoencoder_gpu import *
from linearDecoder import *
from softmax import *
from optimized_mlp_gpu import *
import time
import scipy.io
import gnumpy as gpu

def multilayer_feature_learning(data, inputSize, l1Size, l2Size, l3Size, sparsityParam, lambda_val, beta):
    print "Now starting feature abstraction..."
    num_input = inputSize
    num_hidden_L1 = l1Size
    num_hidden_L2 = l2Size
    num_hidden_L3 = l3Size
    num_output_L1 = inputSize
    num_output_L2 = num_hidden_L1
    num_output_L3 = num_hidden_L2
    sparsityParam = sparsityParam
    lambda_val = lambda_val
    beta = beta
    inputs = gpu.garray(data)
    r = gpu.sqrt(6)/gpu.sqrt(num_hidden_L1+num_input+1)
    weights1_L1 = (gpu.rand(num_hidden_L1,num_input+1))*2*r-r
    weights2_L1 = (gpu.rand(num_output_L1,num_hidden_L1+1))*2*r-r
    num_weights1_L1 = (num_input+1)*num_hidden_L1
    num_weights2_L1 = (num_hidden_L1+1)*num_output_L1
    #weights1_L1 = reshape(weights1_L1, num_weights1_L1)
    weights1_L1 = weights1_L1.reshape(num_weights1_L1)
    #weights2_L1 = reshape(weights2_L1, num_weights2_L1)
    weights2_L1 = weights2_L1.reshape(num_weights2_L1)
    weights_L1 = hstack((weights1_L1.as_numpy_array(),weights2_L1.as_numpy_array()))
    print "Level 1 Abstraction Starting...."
    args = (num_input, num_hidden_L1, num_output_L1, inputs, lambda_val, sparsityParam, beta)
    opttheta_L1 = optimize.fmin_l_bfgs_b(costfunc_gpu, weights_L1, fprime=grad_costfunc_gpu, args=args, maxiter=400)
    weights_L1 = gpu.garray(opttheta_L1[0])
    #weights1_L1 = reshape(weights_L1[0:num_weights1_L1],(num_hidden_L1,num_input+1))
    weights1_L1 = weights_L1[0:num_weights1_L1].reshape((num_hidden_L1,num_input+1))
    #weights2_L1 = reshape(weights_L1[num_weights1_L1:shape(weights_L1)[0]],(num_hidden_L2,num_hidden_L1+1))
    weights2_L1 = weights_L1[num_weights1_L1:shape(weights_L1)[0]].reshape((num_output_L1,num_hidden_L1+1))
    scipy.io.savemat('MINSTLevel1.mat', mdict={'learntFeaturesL1_1': weights1_L1.as_numpy_array(), 'learntFeaturesL1_2': weights2_L1.as_numpy_array()})
    L1_activation = feedforward(weights1_L1, inputs)
    del weights_L1
    del weights1_L1
    del weights2_L1
    gpu.free_reuse_cache()
    v = gpu.sqrt(6)/gpu.sqrt(num_hidden_L2+num_hidden_L1+1)
    weights1_L2 = (gpu.rand(num_hidden_L2,num_hidden_L1+1))*2*v-v
    weights2_L2 = (gpu.rand(num_output_L2,num_hidden_L2+1))*2*v-v
    num_weights1_L2 = (num_hidden_L1+1)*num_hidden_L2
    num_weights2_L2 = (num_hidden_L2+1)*num_output_L2
    #weights1_L2 = reshape(weights1_L2, num_weights1_L2)
    weights1_L2 = weights1_L2.reshape(num_weights1_L2)
    #weights2_L2 = reshape(weights2_L2, num_weights2_L2)
    weights2_L2 = weights2_L2.reshape(num_weights2_L2)
    weights_L2 = hstack((weights1_L2.as_numpy_array(),weights2_L2.as_numpy_array()))
    args = (num_hidden_L1, num_hidden_L2, num_output_L2, L1_activation, lambda_val, sparsityParam, beta)
    print "Level 2 Abstraction Starting...."
    opttheta_L2 = optimize.fmin_l_bfgs_b(costfunc_gpu, weights_L2, fprime=grad_costfunc_gpu, args=args, maxiter=400)
    weights_L2 = gpu.garray(opttheta_L2[0])
    #weights1_L2 = reshape(weights_L2[0:num_weights1_L2],(num_hidden_L2,num_hidden_L1+1))
    weights1_L2 = weights_L2[0:num_weights1_L2].reshape((num_hidden_L2,num_hidden_L1+1))
    weights2_L2 = weights_L2[num_weights1_L2:shape(weights_L2)[0]].reshape((num_output_L2,num_hidden_L2+1))
    scipy.io.savemat('MINSTLevel2.mat', mdict={'learntFeaturesL2_1': weights1_L2.as_numpy_array(),'learntFeaturesL2_2': weights2_L2.as_numpy_array()})
    L2_activation = feedforward(weights1_L2, L1_activation)
    del weights_L2
    del weights1_L2
    del weights2_L2
    gpu.free_reuse_cache()
    u = gpu.sqrt(6)/gpu.sqrt(num_hidden_L3+num_hidden_L2+1)
    weights1_L3 = (gpu.rand(num_hidden_L3,num_hidden_L2+1))*2*u-u
    weights2_L3 = (gpu.rand(num_output_L3,num_hidden_L3+1))*2*u-u
    num_weights1_L3 = (num_hidden_L2+1)*num_hidden_L3
    num_weights2_L3 = (num_hidden_L3+1)*num_output_L3
    #weights1_L3 = reshape(weights1_L3, num_weights1_L3)
    weights1_L3 = weights1_L3.reshape(num_weights1_L3)
    #weights2_L3 = reshape(weights2_L3, num_weights2_L3)
    weights2_L3 = weights2_L3.reshape(num_weights2_L3)
    weights_L3 = hstack((weights1_L3.as_numpy_array(),weights2_L3.as_numpy_array()))
    args = (num_hidden_L2, num_hidden_L3, num_output_L3, L2_activation, lambda_val, sparsityParam, beta)
    print "Level 3 Abstraction Starting...."
    opttheta_L3 = optimize.fmin_l_bfgs_b(costfunc_gpu, weights_L3, fprime=grad_costfunc_gpu, args=args, maxiter=400)
    weights_L3 = gpu.garray(opttheta_L3[0])
    #weights1_L3 = reshape(weights_L3[0:num_weights1_L3],(num_hidden_L3,num_hidden_L2+1))
    weights1_L3 = weights_L3[0:num_weights1_L3].reshape((num_hidden_L3,num_hidden_L2+1))
    weights2_L3 = weights_L3[num_weights1_L3:shape(weights_L3)[0]].reshape((num_output_L3,num_hidden_L3+1))
    scipy.io.savemat('MINSTLevel3.mat', mdict={'learntFeaturesL3_1': weights1_L3.as_numpy_array(),'learntFeaturesL3_2': weights2_L3.as_numpy_array()})
    L3_activation = feedforward(weights1_L3, L2_activation)
    del weights_L3
    del weights1_L3
    del weights2_L3
    gpu.free_reuse_cache()
    print "Abstraction completed."
    return L3_activation

def feedforward(theta, data):
    nData = shape(data)[1]
    x = gpu.concatenate((gpu.ones((1,nData)), data), axis = 0)
    hidden_sum = gpu.dot(theta, x)
    hidden_activation = hidden_sum.logistic()
    return hidden_activation

def fine_tuning_cost_cpu(x, *args):
    inputSize, l1Size, l2Size, l3Size, l4Size, l5Size, lambda_val, inputs = args
    num_weights_L1 = l1Size * (inputSize + 1)
    num_weights_L2 = l2Size * (l1Size + 1)
    num_weights_L3 = l3Size * (l2Size + 1)
    num_weights_L4 = l4Size * (l3Size + 1)
    num_weights_L5 = l5Size * (l4Size + 1)
    #num_weights_L6 = inputSize * (l5Size + 1)
    weights1 = reshape(x[0:num_weights_L1], (l1Size, inputSize + 1))
    weights2 = reshape(x[num_weights_L1:num_weights_L1+num_weights_L2], (l2Size, l1Size + 1))
    weights3 = reshape(x[num_weights_L1+num_weights_L2:num_weights_L1+num_weights_L2+num_weights_L3], (l3Size, l2Size + 1))
    weights4 = reshape(x[num_weights_L1+num_weights_L2+num_weights_L3:num_weights_L1+num_weights_L2+num_weights_L3+num_weights_L4], (l4Size, l3Size + 1))
    weights5 = reshape(x[num_weights_L1+num_weights_L2+num_weights_L3+num_weights_L4:num_weights_L1+num_weights_L2+num_weights_L3+num_weights_L4+num_weights_L5], (l5Size, l4Size + 1))
    weights6 = reshape(x[num_weights_L1+num_weights_L2+num_weights_L3+num_weights_L4+num_weights_L5:shape(x)[0]], (inputSize, l5Size+1))
    nData = shape(inputs)[1]
    x = concatenate((ones((1,nData)), inputs), axis = 0)
    hidden1_sum = dot(weights1, x)
    hidden1_activation = 1/(1 + exp(-hidden1_sum))
    hidden1_activation = concatenate((ones((1,nData)), hidden1_activation), axis = 0)
    hidden2_sum = dot(weights2, hidden1_activation)
    hidden2_activation = 1/(1 + exp(-hidden2_sum))
    hidden2_activation = concatenate((ones((1,nData)), hidden2_activation), axis = 0)
    hidden3_sum = dot(weights3, hidden2_activation)
    hidden3_activation = 1/(1 + exp(-hidden3_sum))
    hidden3_activation = concatenate((ones((1,nData)), hidden3_activation), axis = 0)
    hidden4_sum = dot(weights4, hidden3_activation)
    hidden4_activation = 1/(1 + exp(-hidden4_sum))
    hidden4_activation = concatenate((ones((1,nData)), hidden4_activation), axis = 0)
    hidden5_sum = dot(weights5, hidden4_activation)
    hidden5_activation = 1/(1 + exp(-hidden5_sum))
    hidden5_activation = concatenate((ones((1,nData)), hidden5_activation), axis = 0)
    output_sum = dot(weights6, hidden5_activation)
    outputs = 1/(1 + exp(-output_sum))
    regularized_penalty1 = weights1[:,1:shape(weights1)[1]]
    regularized_penalty2 = weights2[:,1:shape(weights2)[1]]
    regularized_penalty3 = weights3[:,1:shape(weights3)[1]]
    regularized_penalty4 = weights4[:,1:shape(weights4)[1]]
    regularized_penalty5 = weights5[:,1:shape(weights5)[1]]
    regularized_penalty6 = weights6[:,1:shape(weights6)[1]]
    regularized_penalty1 = regularized_penalty1 ** 2
    regularized_penalty2 = regularized_penalty2 ** 2
    regularized_penalty3 = regularized_penalty3 ** 2
    regularized_penalty4 = regularized_penalty4 ** 2
    regularized_penalty5 = regularized_penalty5 ** 2
    regularized_penalty6 = regularized_penalty6 ** 2
    output_target_diff = (outputs - inputs)**2
    cost = sum(output_target_diff)/(2*nData) + 0.5 * lambda_val * (sum(regularized_penalty1) + sum(regularized_penalty2) + sum(regularized_penalty3) + sum(regularized_penalty4) + sum(regularized_penalty5) + sum(regularized_penalty6) )
    print 'Fine Tuning Cost: ', cost
    return cost

def fine_tuning_cost_gpu(x, *args):
    inputSize, l1Size, l2Size, l3Size, l4Size, l5Size, lambda_val, inputs = args
    num_weights_L1 = l1Size * (inputSize + 1)
    num_weights_L2 = l2Size * (l1Size + 1)
    num_weights_L3 = l3Size * (l2Size + 1)
    num_weights_L4 = l4Size * (l3Size + 1)
    num_weights_L5 = l5Size * (l4Size + 1)
    #num_weights_L6 = inputSize * (l5Size + 1)
    x = gpu.garray(x)
    inputs = gpu.garray(inputs)
    #weights1 = reshape(x[0:num_weights_L1], (l1Size, inputSize + 1))
    weights1 = x[0:num_weights_L1].reshape((l1Size, inputSize + 1))
    #weights2 = reshape(x[num_weights_L1:num_weights_L1+num_weights_L2], (l2Size, l1Size + 1))
    weights2 = x[num_weights_L1:num_weights_L1+num_weights_L2].reshape((l2Size, l1Size + 1))
    #weights3 = reshape(x[num_weights_L1+num_weights_L2:num_weights_L1+num_weights_L2+num_weights_L3], (l3Size, l2Size + 1))
    weights3 = x[num_weights_L1+num_weights_L2:num_weights_L1+num_weights_L2+num_weights_L3].reshape((l3Size, l2Size + 1))
    #weights4 = reshape(x[num_weights_L1+num_weights_L2+num_weights_L3:num_weights_L1+num_weights_L2+num_weights_L3+num_weights_L4], (l4Size, l3Size + 1))
    weights4 = x[num_weights_L1+num_weights_L2+num_weights_L3:num_weights_L1+num_weights_L2+num_weights_L3+num_weights_L4].reshape((l4Size, l3Size + 1))
    #weights5 = reshape(x[num_weights_L1+num_weights_L2+num_weights_L3+num_weights_L4:num_weights_L1+num_weights_L2+num_weights_L3+num_weights_L4+num_weights_L5], (l5Size, l4Size + 1))
    weights5 = x[num_weights_L1+num_weights_L2+num_weights_L3+num_weights_L4:num_weights_L1+num_weights_L2+num_weights_L3+num_weights_L4+num_weights_L5].reshape((l5Size, l4Size + 1))
    #weights6 = reshape(x[num_weights_L1+num_weights_L2+num_weights_L3+num_weights_L4+num_weights_L5:shape(x)[0]], (inputSize, l5Size+1))
    weights6 = x[num_weights_L1+num_weights_L2+num_weights_L3+num_weights_L4+num_weights_L5:shape(x)[0]].reshape((inputSize, l5Size+1))
    nData = shape(inputs)[1]
    x = gpu.concatenate((gpu.ones((1,nData)), inputs), axis = 0)
    hidden1_sum = gpu.dot(weights1, x)
    hidden1_activation = hidden1_sum.logistic()
    hidden1_activation = gpu.concatenate((gpu.ones((1,nData)), hidden1_activation), axis = 0)
    hidden2_sum = gpu.dot(weights2, hidden1_activation)
    hidden2_activation = hidden2_sum.logistic()
    hidden2_activation = gpu.concatenate((gpu.ones((1,nData)), hidden2_activation), axis = 0)
    hidden3_sum = gpu.dot(weights3, hidden2_activation)
    hidden3_activation = hidden3_sum.logistic()
    hidden3_activation = gpu.concatenate((gpu.ones((1,nData)), hidden3_activation), axis = 0)
    hidden4_sum = gpu.dot(weights4, hidden3_activation)
    hidden4_activation = hidden4_sum.logistic()
    hidden4_activation = gpu.concatenate((gpu.ones((1,nData)), hidden4_activation), axis = 0)
    hidden5_sum = gpu.dot(weights5, hidden4_activation)
    hidden5_activation = hidden5_sum.logistic()
    hidden5_activation = gpu.concatenate((gpu.ones((1,nData)), hidden5_activation), axis = 0)
    output_sum = gpu.dot(weights6, hidden5_activation)
    outputs = output_sum.logistic()
    regularized_penalty4 = weights4[:,1:shape(weights4)[1]]
    regularized_penalty5 = weights5[:,1:shape(weights5)[1]]
    regularized_penalty6 = weights6[:,1:shape(weights6)[1]]
    regularized_penalty4 = regularized_penalty4 ** 2
    regularized_penalty5 = regularized_penalty5 ** 2
    regularized_penalty6 = regularized_penalty6 ** 2
    output_target_diff = (outputs - inputs)**2
    cost = gpu.sum(output_target_diff)/(2*nData) + 0.5 * lambda_val * (gpu.sum(regularized_penalty4) + gpu.sum(regularized_penalty5) + gpu.sum(regularized_penalty6))
    print 'Fine Tuning Cost: ', cost
    return cost



def fine_tuning_grad_cpu(x, *args):
    inputSize, l1Size, l2Size, l3Size, l4Size, l5Size, lambda_val, inputs = args
    num_weights_L1 = l1Size * (inputSize + 1)
    num_weights_L2 = l2Size * (l1Size + 1)
    num_weights_L3 = l3Size * (l2Size + 1)
    num_weights_L4 = l4Size * (l3Size + 1)
    num_weights_L5 = l5Size * (l4Size + 1)
    num_weights_L6 = inputSize * (l5Size + 1)
    weights1 = reshape(x[0:num_weights_L1], (l1Size, inputSize + 1))
    weights2 = reshape(x[num_weights_L1:num_weights_L1+num_weights_L2], (l2Size, l1Size + 1))
    weights3 = reshape(x[num_weights_L1+num_weights_L2:num_weights_L1+num_weights_L2+num_weights_L3], (l3Size, l2Size + 1))
    weights4 = reshape(x[num_weights_L1+num_weights_L2+num_weights_L3:num_weights_L1+num_weights_L2+num_weights_L3+num_weights_L4], (l4Size, l3Size + 1))
    weights5 = reshape(x[num_weights_L1+num_weights_L2+num_weights_L3+num_weights_L4:num_weights_L1+num_weights_L2+num_weights_L3+num_weights_L4+num_weights_L5], (l5Size, l4Size + 1))
    weights6 = reshape(x[num_weights_L1+num_weights_L2+num_weights_L3+num_weights_L4+num_weights_L5:shape(x)[0]], (inputSize, l5Size+1))
    nData = shape(inputs)[1]
    x = concatenate((ones((1,nData)), inputs), axis = 0)
    hidden1_sum = dot(weights1, x)
    hidden1_activation = 1/(1 + exp(-hidden1_sum))
    hidden1_activation = concatenate((ones((1,nData)), hidden1_activation), axis = 0)
    hidden2_sum = dot(weights2, hidden1_activation)
    hidden2_activation = 1/(1 + exp(-hidden2_sum))
    hidden2_activation = concatenate((ones((1,nData)), hidden2_activation), axis = 0)
    hidden3_sum = dot(weights3, hidden2_activation)
    hidden3_activation = 1/(1 + exp(-hidden3_sum))
    hidden3_activation = concatenate((ones((1,nData)), hidden3_activation), axis = 0)
    hidden4_sum = dot(weights4, hidden3_activation)
    hidden4_activation = 1/(1 + exp(-hidden4_sum))
    hidden4_activation = concatenate((ones((1,nData)), hidden4_activation), axis = 0)
    hidden5_sum = dot(weights5, hidden4_activation)
    hidden5_activation = 1/(1 + exp(-hidden5_sum))
    hidden5_activation = concatenate((ones((1,nData)), hidden5_activation), axis = 0)
    output_sum = dot(weights6, hidden5_activation)
    outputs = 1/(1 + exp(-output_sum))
    weights1_grad = zeros(shape(weights1))
    weights2_grad = zeros(shape(weights2))
    weights3_grad = zeros(shape(weights3))
    weights4_grad = zeros(shape(weights4))
    weights5_grad = zeros(shape(weights5))
    weights6_grad = zeros(shape(weights6))
    a = multiply(multiply((outputs - inputs), outputs), (1-outputs))
    weights6_grad += dot(a, transpose(hidden5_activation))
    b_temp = dot(transpose(weights6),a)
    b = multiply(multiply(b_temp,hidden5_activation),(1-hidden5_activation))
    delta2 = dot(b, transpose(hidden4_activation))
    weights5_grad += delta2[1:shape(delta2)[0], :]
    c_temp = dot(transpose(weights5), b[1:shape(b)[0], :])
    c = multiply(multiply(c_temp,hidden4_activation),(1-hidden4_activation))
    delta3 = dot(c, transpose(hidden3_activation))
    weights4_grad += delta3[1:shape(delta3)[0], :]
    d_temp = dot(transpose(weights4), c[1:shape(c)[0], :])
    d = multiply(multiply(d_temp,hidden3_activation),(1-hidden3_activation))
    delta4 = dot(d, transpose(hidden2_activation))
    weights3_grad += delta4[1:shape(delta4)[0], :]
    e_temp = dot(transpose(weights3), d[1:shape(d)[0], :])
    e = multiply(multiply(e_temp,hidden2_activation),(1-hidden2_activation))
    delta5 = dot(e, transpose(hidden1_activation))
    weights2_grad += delta5[1:shape(delta5)[0], :]
    f_temp = dot(transpose(weights2), e[1:shape(e)[0], :])
    f = multiply(multiply(f_temp,hidden1_activation),(1-hidden1_activation))
    delta6 = dot(f, transpose(x))
    weights1_grad += delta6[1:shape(delta6)[0], :]
    weights1_grad = weights1_grad/nData
    weights2_grad = weights2_grad/nData
    weights3_grad = weights3_grad/nData
    weights4_grad = weights4_grad/nData
    weights5_grad = weights5_grad/nData
    weights6_grad = weights6_grad/nData
    weights4_grad[:,1:shape(weights4_grad)[1]] = weights4_grad[:,1:shape(weights4_grad)[1]] + weights4[:,1:shape(weights4)[1]] * lambda_val
    weights5_grad[:,1:shape(weights5_grad)[1]] = weights5_grad[:,1:shape(weights5_grad)[1]] + weights5[:,1:shape(weights5)[1]] * lambda_val
    weights6_grad[:,1:shape(weights6_grad)[1]] = weights6_grad[:,1:shape(weights6_grad)[1]] + weights6[:,1:shape(weights6)[1]] * lambda_val
    weights1_grad = reshape(weights1_grad, num_weights_L1)
    weights2_grad = reshape(weights2_grad, num_weights_L2)
    weights3_grad = reshape(weights3_grad, num_weights_L3)
    weights4_grad = reshape(weights4_grad, num_weights_L4)
    weights5_grad = reshape(weights5_grad, num_weights_L5)
    weights6_grad = reshape(weights6_grad, num_weights_L6)
    return hstack((weights1_grad,weights2_grad,weights3_grad,weights4_grad,weights5_grad,weights6_grad))

def fine_tuning_grad_gpu(x, *args):
    inputSize, l1Size, l2Size, l3Size, l4Size, l5Size, lambda_val, inputs = args
    num_weights_L1 = l1Size * (inputSize + 1)
    num_weights_L2 = l2Size * (l1Size + 1)
    num_weights_L3 = l3Size * (l2Size + 1)
    num_weights_L4 = l4Size * (l3Size + 1)
    num_weights_L5 = l5Size * (l4Size + 1)
    num_weights_L6 = inputSize * (l5Size + 1)
    x = gpu.garray(x)
    inputs = gpu.garray(inputs)
    #weights1 = reshape(x[0:num_weights_L1], (l1Size, inputSize + 1))
    weights1 = x[0:num_weights_L1].reshape((l1Size, inputSize + 1))
    #weights2 = reshape(x[num_weights_L1:num_weights_L1+num_weights_L2], (l2Size, l1Size + 1))
    weights2 = x[num_weights_L1:num_weights_L1+num_weights_L2].reshape((l2Size, l1Size + 1))
    #weights3 = reshape(x[num_weights_L1+num_weights_L2:num_weights_L1+num_weights_L2+num_weights_L3], (l3Size, l2Size + 1))
    weights3 = x[num_weights_L1+num_weights_L2:num_weights_L1+num_weights_L2+num_weights_L3].reshape((l3Size, l2Size + 1))
    #weights4 = reshape(x[num_weights_L1+num_weights_L2+num_weights_L3:num_weights_L1+num_weights_L2+num_weights_L3+num_weights_L4], (l4Size, l3Size + 1))
    weights4 = x[num_weights_L1+num_weights_L2+num_weights_L3:num_weights_L1+num_weights_L2+num_weights_L3+num_weights_L4].reshape((l4Size, l3Size + 1))
    #weights5 = reshape(x[num_weights_L1+num_weights_L2+num_weights_L3+num_weights_L4:num_weights_L1+num_weights_L2+num_weights_L3+num_weights_L4+num_weights_L5], (l5Size, l4Size + 1))
    weights5 = x[num_weights_L1+num_weights_L2+num_weights_L3+num_weights_L4:num_weights_L1+num_weights_L2+num_weights_L3+num_weights_L4+num_weights_L5].reshape((l5Size, l4Size + 1))
    #weights6 = reshape(x[num_weights_L1+num_weights_L2+num_weights_L3+num_weights_L4+num_weights_L5:shape(x)[0]], (inputSize, l5Size+1))
    weights6 = x[num_weights_L1+num_weights_L2+num_weights_L3+num_weights_L4+num_weights_L5:shape(x)[0]].reshape((inputSize, l5Size+1))
    nData = shape(inputs)[1]
    x = gpu.concatenate((gpu.ones((1,nData)), inputs), axis = 0)
    hidden1_sum = gpu.dot(weights1, x)
    hidden1_activation = hidden1_sum.logistic()
    hidden1_activation = gpu.concatenate((gpu.ones((1,nData)), hidden1_activation), axis = 0)
    hidden2_sum = gpu.dot(weights2, hidden1_activation)
    hidden2_activation = hidden2_sum.logistic()
    hidden2_activation = gpu.concatenate((gpu.ones((1,nData)), hidden2_activation), axis = 0)
    hidden3_sum = gpu.dot(weights3, hidden2_activation)
    hidden3_activation = hidden3_sum.logistic()
    hidden3_activation = gpu.concatenate((gpu.ones((1,nData)), hidden3_activation), axis = 0)
    hidden4_sum = gpu.dot(weights4, hidden3_activation)
    hidden4_activation = hidden4_sum.logistic()
    hidden4_activation = gpu.concatenate((gpu.ones((1,nData)), hidden4_activation), axis = 0)
    hidden5_sum = gpu.dot(weights5, hidden4_activation)
    hidden5_activation = hidden5_sum.logistic()
    hidden5_activation = gpu.concatenate((gpu.ones((1,nData)), hidden5_activation), axis = 0)
    output_sum = gpu.dot(weights6, hidden5_activation)
    outputs = output_sum.logistic()
    weights1_grad = gpu.zeros(shape(weights1))
    weights2_grad = gpu.zeros(shape(weights2))
    weights3_grad = gpu.zeros(shape(weights3))
    weights4_grad = gpu.zeros(shape(weights4))
    weights5_grad = gpu.zeros(shape(weights5))
    weights6_grad = gpu.zeros(shape(weights6))
    #a = multiply(multiply((outputs - inputs), outputs), (1-outputs))
    a = (outputs - inputs) * outputs * (1-outputs)
    weights6_grad += gpu.dot(a, gpu.garray(transpose(hidden5_activation.as_numpy_array())))
    b_temp = gpu.dot(gpu.garray(transpose(weights6.as_numpy_array())),a)
    #b = multiply(multiply(b_temp,hidden5_activation),(1-hidden5_activation))
    b = (b_temp*hidden5_activation) * (1-hidden5_activation)
    delta2 = gpu.dot(b, gpu.garray(transpose(hidden4_activation.as_numpy_array())))
    weights5_grad += delta2[1:shape(delta2)[0], :]
    c_temp = gpu.dot(gpu.garray(transpose(weights5.as_numpy_array())), b[1:shape(b)[0], :])
    #c = multiply(multiply(c_temp,hidden4_activation),(1-hidden4_activation))
    c = (c_temp*hidden4_activation)*(1-hidden4_activation)
    delta3 = gpu.dot(c, gpu.garray(transpose(hidden3_activation.as_numpy_array())))
    weights4_grad += delta3[1:shape(delta3)[0], :]
    d_temp = gpu.dot(gpu.garray(transpose(weights4.as_numpy_array())), c[1:shape(c)[0], :])
    #d = multiply(multiply(d_temp,hidden3_activation),(1-hidden3_activation))
    d = (d_temp*hidden3_activation)*(1-hidden3_activation)
    delta4 = gpu.dot(d, gpu.garray(transpose(hidden2_activation.as_numpy_array())))
    weights3_grad += delta4[1:shape(delta4)[0], :]
    e_temp = gpu.dot(gpu.garray(transpose(weights3.as_numpy_array())), d[1:shape(d)[0], :])
    #e = multiply(multiply(e_temp,hidden2_activation),(1-hidden2_activation))
    e = (e_temp*hidden2_activation)*(1-hidden2_activation)
    delta5 = gpu.dot(e, gpu.garray(transpose(hidden1_activation.as_numpy_array())))
    weights2_grad += delta5[1:shape(delta5)[0], :]
    f_temp = gpu.dot(gpu.garray(transpose(weights2.as_numpy_array())), e[1:shape(e)[0], :])
    #f = multiply(multiply(f_temp,hidden1_activation),(1-hidden1_activation))
    f = (f_temp*hidden1_activation)*(1-hidden1_activation)
    delta6 = gpu.dot(f, gpu.garray(transpose(x.as_numpy_array())))
    weights1_grad += delta6[1:shape(delta6)[0], :]
    weights1_grad = weights1_grad/nData
    weights2_grad = weights2_grad/nData
    weights3_grad = weights3_grad/nData
    weights4_grad = weights4_grad/nData
    weights5_grad = weights5_grad/nData
    weights6_grad = weights6_grad/nData
    weights4_grad[:,1:shape(weights4_grad)[1]] = weights4_grad[:,1:shape(weights4_grad)[1]] + weights4[:,1:shape(weights4)[1]] * lambda_val
    weights5_grad[:,1:shape(weights5_grad)[1]] = weights5_grad[:,1:shape(weights5_grad)[1]] + weights5[:,1:shape(weights5)[1]] * lambda_val
    weights6_grad[:,1:shape(weights6_grad)[1]] = weights6_grad[:,1:shape(weights6_grad)[1]] + weights6[:,1:shape(weights6)[1]] * lambda_val
    weights1_grad = reshape(weights1_grad.as_numpy_array(), num_weights_L1)
    weights2_grad = reshape(weights2_grad.as_numpy_array(), num_weights_L2)
    weights3_grad = reshape(weights3_grad.as_numpy_array(), num_weights_L3)
    weights4_grad = reshape(weights4_grad.as_numpy_array(), num_weights_L4)
    weights5_grad = reshape(weights5_grad.as_numpy_array(), num_weights_L5)
    weights6_grad = reshape(weights6_grad.as_numpy_array(), num_weights_L6)
    return hstack((weights1_grad,weights2_grad,weights3_grad,weights4_grad,weights5_grad,weights6_grad))

def running(inputData):
# multilayer_feature_learning(data, inputSize, l1Size, l2Size, l3Size, sparsityParam, lambda_val, beta)
#    inputSize = shape(win_data)[0]
    inputs = inputData
    inputSize = 96 * 96
    l1Size = 10000
    l2Size = 1024
    l3Size = 196
    sparsityParam = 0.1
    lambda_val = 3e-3
    beta = 3
    multilayer_feature_learning(inputs, inputSize, l1Size, l2Size, l3Size, sparsityParam, lambda_val, beta)
    weights1 = scipy.io.loadmat('MINSTLevel1.mat')['learntFeaturesL1_1']
    weights2 = scipy.io.loadmat('MINSTLevel2.mat')['learntFeaturesL2_1']
    weights3 = scipy.io.loadmat('MINSTLevel3.mat')['learntFeaturesL3_1']
    weights4 = scipy.io.loadmat('MINSTLevel3.mat')['learntFeaturesL3_2']
    weights5 = scipy.io.loadmat('MINSTLevel2.mat')['learntFeaturesL2_2']
    weights6 = scipy.io.loadmat('MINSTLevel1.mat')['learntFeaturesL1_2']
    gpu.free_reuse_cache()
# fine tuning phase
    print "Entering Final Stage: Fine Tuning the entire network..."
    num_input = inputSize
    num_hidden1 = l1Size
    num_hidden2 = l2Size
    num_hidden3 = l3Size
    num_hidden4 = l2Size
    num_hidden5 = l1Size
    num_output = num_input
    num_weights1 = (num_input+1)*num_hidden1
    num_weights2 = (num_hidden1+1)*num_hidden2
    num_weights3 = (num_hidden2+1)*num_hidden3
    num_weights4 = (num_hidden3+1)*num_hidden4
    num_weights5 = (num_hidden4+1)*num_hidden5
    num_weights6 = (num_hidden5+1)*num_output
    weights1 = reshape(weights1, num_weights1)
    weights2 = reshape(weights2, num_weights2)
    weights3 = reshape(weights3, num_weights3)
    weights4 = reshape(weights4, num_weights4)
    weights5 = reshape(weights5, num_weights5)
    weights6 = reshape(weights6, num_weights6)
    weights = hstack((weights1,weights2,weights3,weights4,weights5,weights6))
# inputSize, l1Size, l2Size, l3Size, l4Size, l5Size, lambda_val, inputs = args
    print "Fine Tuning Starting..."
    stepSize = 14702
    for i in range(int(shape(inputs)[1]/stepSize)):
        print "Batch:", i
        data = inputs[:,i*stepSize:(i+1)*stepSize]
	args = (num_input, num_hidden1, num_hidden2, num_hidden3, num_hidden4, num_hidden5, lambda_val, data)
    	opttheta = optimize.fmin_l_bfgs_b(fine_tuning_cost_gpu, weights, fprime=fine_tuning_grad_gpu, args=args, maxiter=400)
    	weights = opttheta[0]
	del opttheta
	gpu.free_reuse_cache()
    weights1 = reshape(weights[0:num_weights1], (l1Size, inputSize + 1))
    weights2 = reshape(weights[num_weights1:num_weights1+num_weights2], (l2Size, l1Size + 1))
    weights3 = reshape(weights[num_weights1+num_weights2:num_weights1+num_weights2+num_weights3], (l3Size, l2Size + 1))
    weights4 = reshape(weights[num_weights1+num_weights2+num_weights3:num_weights1+num_weights2+num_weights3+num_weights4], (num_hidden4, num_hidden3 + 1))
    weights5 = reshape(weights[num_weights1+num_weights2+num_weights3+num_weights4:num_weights1+num_weights2+num_weights3+num_weights4+num_weights5], (num_hidden5, num_hidden4 + 1))
    weights6 = reshape(weights[num_weights1+num_weights2+num_weights3+num_weights4+num_weights5:shape(weights)[0]], (inputSize, num_hidden5+1))
    scipy.io.savemat('MINST_FineTuned_features.mat', mdict={'learntFeaturesL1': weights1,'learntFeaturesL2': weights2, 'learntFeaturesL3': weights3, 'learntFeaturesL4': weights4, 'learntFeaturesL5': weights5, 'learntFeaturesL6': weights6})
    trainData = scipy.io.loadmat('trainData.mat')['trainData']
    train_weights1 = reshape(weights1, num_weights1)
    train_weights2 = reshape(weights2, num_weights2)
    train_weights3 = reshape(weights3, num_weights3)
    train_weights4 = reshape(weights4, num_weights4)
    train_weights5 = reshape(weights5, num_weights5)
    train_weights6 = reshape(weights6, num_weights6)
    train_weights = hstack((train_weights1,train_weights2,train_weights3,train_weights4,train_weights5,train_weights6))
    args = (num_input, num_hidden1, num_hidden2, num_hidden3, num_hidden4, num_hidden5, lambda_val, trainData)
    opttheta = optimize.fmin_l_bfgs_b(fine_tuning_cost_gpu, train_weights, fprime=fine_tuning_grad_gpu, args=args, maxiter=400)
    train_weights = opttheta[0]
    del opttheta
    gpu.free_reuse_cache()
    train_weights1 = reshape(train_weights[0:num_weights1], (l1Size, inputSize + 1))
    train_weights2 = reshape(train_weights[num_weights1:num_weights1+num_weights2], (l2Size, l1Size + 1))
    train_weights3 = reshape(train_weights[num_weights1+num_weights2:num_weights1+num_weights2+num_weights3], (l3Size, l2Size + 1))
    train_weights4 = reshape(train_weights[num_weights1+num_weights2+num_weights3:num_weights1+num_weights2+num_weights3+num_weights4], (num_hidden4, num_hidden3 + 1))
    train_weights5 = reshape(train_weights[num_weights1+num_weights2+num_weights3+num_weights4:num_weights1+num_weights2+num_weights3+num_weights4+num_weights5], (num_hidden5, num_hidden4 + 1))
    train_weights6 = reshape(train_weights[num_weights1+num_weights2+num_weights3+num_weights4+num_weights5:shape(weights)[0]], (inputSize, num_hidden5+1))
    testData = scipy.io.loadmat('testData.mat')['testData']
    nData = shape(testData)[1]
    x = concatenate((ones((1,nData)), testData), axis = 0)
    hidden1_sum = dot(train_weights1, x)
    hidden1_activation = 1/(1 + exp(-hidden1_sum))
    hidden1_activation = concatenate((ones((1,nData)), hidden1_activation), axis = 0)
    hidden2_sum = dot(train_weights2, hidden1_activation)
    hidden2_activation = 1/(1 + exp(-hidden2_sum))
    hidden2_activation = concatenate((ones((1,nData)), hidden2_activation), axis = 0)
    hidden3_sum = dot(train_weights3, hidden2_activation)
    hidden3_activation = 1/(1 + exp(-hidden3_sum))
    hidden3_activation = concatenate((ones((1,nData)), hidden3_activation), axis = 0)
    hidden4_sum = dot(train_weights4, hidden3_activation)
    hidden4_activation = 1/(1 + exp(-hidden4_sum))
    hidden4_activation = concatenate((ones((1,nData)), hidden4_activation), axis = 0)
    hidden5_sum = dot(train_weights5, hidden4_activation)
    hidden5_activation = 1/(1 + exp(-hidden5_sum))
    hidden5_activation = concatenate((ones((1,nData)), hidden5_activation), axis = 0)
    output_sum = dot(train_weights6, hidden5_activation)
    outputs = 1/(1 + exp(-output_sum))
    return outputs


start = time.clock()
gpu.free_reuse_cache()
all_data = scipy.io.loadmat('unlabelData_for_autoencoder')
unlabeledData = all_data['unlabeledData']
recovered_image = running(unlabeledData)
end = time.clock()
print "Total running time:", end - start



