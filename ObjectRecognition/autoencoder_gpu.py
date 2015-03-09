# -*- coding: utf-8 -*-
"""
Created on Wed Nov 06 01:29:36 2013

@author: Michael
"""

from numpy import *
import scipy.io
from scipy import optimize
import time

def costfunc(x, *args):
    num_input,num_hidden,num_output,inputs,lambda_val,sparsityParam,beta = args
    num_weights1 = (num_input+1)*num_hidden
    weights1 = reshape(x[0:num_weights1],(num_hidden,num_input+1))
    weights2 = reshape(x[num_weights1:shape(x)[0]], (num_output,num_hidden+1))
    nData = shape(inputs)[1]
    x = concatenate((ones((1,nData)), inputs), axis = 0)
    hidden_sum = dot(weights1, x)
    hidden_activation = log(1 + exp(hidden_sum))
    p_avg = sum(hidden_activation,axis=1)/nData
    hidden_activation = concatenate((ones((1,nData)), hidden_activation), axis = 0)
    output_sum = dot(weights2, hidden_activation)
    output = log(1 + exp(output_sum))
    regularized_penalty1 = weights1[:,1:shape(weights1)[1]]
    regularized_penalty2 = weights2[:,1:shape(weights2)[1]]
    regularized_penalty1 = regularized_penalty1 ** 2
    regularized_penalty2 = regularized_penalty2 ** 2
    output_target_diff = (output - inputs)**2
    KL = sum(sparsityParam*log(sparsityParam/p_avg) + (1-sparsityParam)*log((1-sparsityParam)/(1-p_avg)))  
    cost = sum(output_target_diff)/(2*nData) + 0.5 * lambda_val * (sum(regularized_penalty1) + sum(regularized_penalty2)) + beta*KL  
    print 'Autoencoder Cost: ', cost   
    return cost
    
def cost_func_no_optimize(inputs, weights1, weights2, lambda_val, sparsityParam, beta):
    nData = shape(inputs)[1]
    x = concatenate((ones((1,nData)), inputs), axis = 0)
    hidden_sum = dot(weights1, x)
    hidden_activation = 1/(1 + exp(-hidden_sum))
    p_avg = sum(hidden_activation,axis=1)/nData
    hidden_activation = concatenate((ones((1,nData)), hidden_activation), axis = 0)
    output_sum = dot(weights2, hidden_activation)
    output = 1/(1 + exp(-output_sum))
    regularized_penalty1 = weights1[:,1:shape(weights1)[1]]
    regularized_penalty2 = weights2[:,1:shape(weights2)[1]]
    regularized_penalty1 = regularized_penalty1 ** 2
    regularized_penalty2 = regularized_penalty2 ** 2
    output_target_diff = (output - inputs)**2
    KL = sum(sparsityParam*log(sparsityParam/p_avg) + (1-sparsityParam)*log((1-sparsityParam)/(1-p_avg)))  
    cost = sum(output_target_diff)/(2*nData) + 0.5 * lambda_val * (sum(regularized_penalty1) + sum(regularized_penalty2)) + beta*KL  
    return cost, p_avg, output, hidden_activation, x
    
def grad_costfunc(x, *args):
    num_input,num_hidden,num_output,inputs,lambda_val,sparsityParam,beta = args
    num_weights1 = (num_input+1)*num_hidden
    num_weights2 = (num_hidden+1)*num_output
    weights1 = reshape(x[0:num_weights1],(num_hidden,num_input+1))
    weights2 = reshape(x[num_weights1:shape(x)[0]], (num_output,num_hidden+1))
    nData = shape(inputs)[1]
    x = concatenate((ones((1,nData)), inputs), axis = 0)
    hidden_sum = dot(weights1, x)
    hidden_activation = log(1 + exp(hidden_sum))
    p_avg = sum(hidden_activation,axis=1)/nData
    grad_sparse = -1*sparsityParam/p_avg + (1-sparsityParam)/(1-p_avg)
    grad_sparse = append(0,grad_sparse)
    grad_sparse = tile(grad_sparse, (nData, 1))
    grad_sparse = transpose(grad_sparse)
    hidden_activation = concatenate((ones((1,nData)), hidden_activation), axis = 0)
    output_sum = dot(weights2, hidden_activation)
    outputs = log(1 + exp(output_sum))
    weights1_grad = zeros(shape(weights1))
    weights2_grad = zeros(shape(weights2))
    p = multiply((outputs - inputs), 1/(1 + exp(-output_sum)))
    weights2_grad += dot(p, transpose(hidden_activation))
    q_temp = dot(transpose(weights2),p) + beta*grad_sparse
    q = multiply(q_temp,1/(1 + exp(-hidden_sum)))
    delta2 = dot(q, transpose(x))
    weights1_grad += delta2[1:shape(delta2)[0], :]
    weights1_grad = weights1_grad/nData
    weights2_grad = weights2_grad/nData
    weights1_grad[:,1:shape(weights1_grad)[1]] = weights1_grad[:,1:shape(weights1_grad)[1]] + weights1[:,1:shape(weights1)[1]] * lambda_val
    weights2_grad[:,1:shape(weights2_grad)[1]] = weights2_grad[:,1:shape(weights2_grad)[1]] + weights2[:,1:shape(weights2)[1]] * lambda_val
    weights1_grad = reshape(weights1_grad, num_weights1)
    weights2_grad = reshape(weights2_grad, num_weights2)
    return hstack((weights1_grad,weights2_grad))
    
def train_no_optimize(inputs,num_hidden,lambda_val,sparsityParam,momentum,beta,theta,kappa,psi,num_iters):
    nData = shape(inputs)[1]
    num_input = shape(inputs)[0]
    num_output = num_input
    r = sqrt(6)/sqrt(num_hidden+num_input+1)
    weights1 = (random.rand(num_hidden, num_input+1))*2*r-r
    weights2 = (random.rand(num_output, num_hidden+1))*2*r-r
    weights1_grad = zeros(shape(weights1))
    weights2_grad = zeros(shape(weights2))
    past_weights1_change = zeros(shape(weights1))
    past_weights2_change = zeros(shape(weights2))
    past_weights1_grad = zeros(shape(weights1))
    past_weights2_grad = zeros(shape(weights2))
    past_alpha1 = 0.1*ones(shape(weights1))
    past_alpha2 = 0.1*ones(shape(weights2))
    past_f1 = zeros(shape(weights1))
    past_f2 = zeros(shape(weights2))
    for n in range(num_iters):
        cost, p_avg, outputs, hidden_activation, x = cost_func_no_optimize(inputs, weights1, weights2, lambda_val, sparsityParam, beta)
        print "Iteration: ", n+1, " | Cost: ", cost
        grad_sparse = -1*sparsityParam/p_avg + (1-sparsityParam)/(1-p_avg)
        grad_sparse = append(0,grad_sparse)
        grad_sparse = tile(grad_sparse, (nData, 1))
        grad_sparse = transpose(grad_sparse)
        p = multiply(multiply((outputs - inputs), outputs), (1-outputs))
        weights2_grad += dot(p, transpose(hidden_activation))
        q_temp = dot(transpose(weights2),p) + beta*grad_sparse
        q = multiply(multiply(q_temp,hidden_activation),(1-hidden_activation))
        delta2 = dot(q, transpose(x))
        weights1_grad += delta2[1:shape(delta2)[0], :]
        f1 = theta*past_f1 + (1-theta)*past_weights1_grad
        f2 = theta*past_f2 + (1-theta)*past_weights2_grad
        f1_check = multiply(f1,weights1_grad/nData)
        f2_check = multiply(f2,weights2_grad/nData)
        alpha1 = past_alpha1 + kappa * where(f1_check>0,1,0)
        a1temp = psi*where(f1_check<=0,1,0)
        a1temp = where(a1temp==0,1,a1temp)
        alpha1 = multiply(past_alpha1,a1temp)
        alpha2 = past_alpha2 + kappa * where(f2_check>0,1,0)
        a2temp = psi*where(f2_check<=0,1,0)
        a2temp = where(a2temp==0,1,a2temp)
        alpha2 = multiply(past_alpha2,a2temp)
        weights1_grad = weights1_grad/nData
        weights2_grad = weights2_grad/nData
        weights1_grad[:,1:shape(weights1_grad)[1]] = weights1_grad[:,1:shape(weights1_grad)[1]] + weights1[:,1:shape(weights1)[1]] * lambda_val
        weights2_grad[:,1:shape(weights2_grad)[1]] = weights2_grad[:,1:shape(weights2_grad)[1]] + weights2[:,1:shape(weights2)[1]] * lambda_val
        weights1_change = momentum * past_weights1_change - multiply((1-alpha1),weights1_grad)
        weights2_change = momentum * past_weights2_change - multiply((1-alpha2),weights2_grad)
        weights1 += weights1_change
        weights2 += weights2_change
        past_weights1_change = weights1_change
        past_weights2_change = weights2_change
        past_weights1_grad = weights1_grad
        past_weights2_grad = weights2_grad
        past_alpha1 = alpha1
        past_alpha2 = alpha2
        past_f1 = f1
        past_f2 = f2
    return weights1,weights2
    
def checkGradient():
    num_input = 64
    num_hidden = 5
    num_output = 64
    lambda_val = 0.0001
    sparsityParam = 0.01
    beta = 3
    img = sampleImg()
    inputs = img[:,0:10]
    r = sqrt(6)/sqrt(num_hidden+num_input+1)
    weights1 = (random.rand(num_hidden,num_input+1))*2*r-r
    weights2 = (random.rand(num_output,num_hidden+1))*2*r-r
    num_weights1 = (num_input+1)*num_hidden
    num_weights2 = (num_hidden+1)*num_output
    weights1 = reshape(weights1, num_weights1)
    weights2 = reshape(weights2, num_weights2)
    weights = hstack((weights1,weights2))
    args = (num_input,num_hidden,num_output,inputs,lambda_val,sparsityParam,beta)
    numgrad = zeros(size(weights))
    perturb = zeros(size(weights))
    e = 1e-4
    for p in range(size(weights)):
        perturb[p] = e;
        minus_weights = weights - perturb
        plus_weights = weights + perturb
        loss1 = costfunc(minus_weights, *args)
        loss2 = costfunc(plus_weights, *args)
        numgrad[p] = (loss2 - loss1) / (2*e)
        perturb[p] = 0
    grad = grad_costfunc(weights, *args)
    diff = norm(numgrad-grad)/norm(numgrad+grad)
    return diff

def sampleImg():
    pat = scipy.io.loadmat('patches.mat')
    patches = pat['patc']
    return transpose(patches)
    
def autoencoder_train():
    num_input = 8*8
    num_hidden = 25
    num_output = 8*8
    lambda_val = 0.0001
    sparsityParam = 0.01
    beta = 3
    inputs = sampleImg()
    r = sqrt(6)/sqrt(num_hidden+num_input+1)
    weights1 = (random.rand(num_hidden,num_input+1))*2*r-r
    weights2 = (random.rand(num_output,num_hidden+1))*2*r-r
    num_weights1 = (num_input+1)*num_hidden
    num_weights2 = (num_hidden+1)*num_output
    weights1 = reshape(weights1, num_weights1)
    weights2 = reshape(weights2, num_weights2)
    weights = hstack((weights1,weights2))
    args = (num_input,num_hidden,num_output,inputs,lambda_val,sparsityParam,beta)
    opttheta = optimize.fmin_l_bfgs_b(costfunc, weights, fprime=grad_costfunc, args=args, maxiter=400)
    weights = opttheta[0]
    weights1 = reshape(weights[0:num_weights1],(num_hidden,num_input+1))
    weights2 = reshape(weights[num_weights1:shape(weights)[0]], (num_output,num_hidden+1))
    scipy.io.savemat('test.mat', mdict={'weights1': weights1})
    return weights1
    
def autoencoder_no_optimize():
    num_hidden = 25
    lambda_val = 0.0001
    sparsityParam = 0.01
    beta = 3
    momentum = 0.9
    theta = 0.7
    kappa = 0.1
    psi = 0.6
    num_iters = 500
    inputs = sampleImg()
    weights1, weights2 = train_no_optimize(inputs,num_hidden,lambda_val,sparsityParam,momentum,beta,theta,kappa,psi,num_iters)
    scipy.io.savemat('test.mat', mdict={'weights1': weights1})
    return weights1
    
def benchmark_opt():
    start = time.clock()
    autoencoder_train()
    end = time.clock()
    return end - start
    
def benchmark_no_opt():
    start = time.clock()
    autoencoder_no_optimize()
    end = time.clock()
    return end - start