from numpy import *
from scipy import optimize
import scipy.io
import gnumpy as gpu
import time


def costfunc(x, *args):
    num_input,num_hidden,num_output,inputs,lambda_val,sparsityParam,beta = args
    num_weights1 = (num_input+1)*num_hidden
    weights1 = reshape(x[0:num_weights1],(num_hidden,num_input+1))
    weights2 = reshape(x[num_weights1:shape(x)[0]], (num_output,num_hidden+1))
    nData = shape(inputs)[1]
    x = concatenate((ones((1,nData)), inputs), axis = 0)
    hidden_sum = dot(weights1, x)
    hidden_activation = 1 / (1+exp(-hidden_sum))
    p_avg = sum(hidden_activation,axis=1)/nData
    hidden_activation = concatenate((ones((1,nData)), hidden_activation), axis = 0)
    output = dot(weights2, hidden_activation)
    regularized_penalty1 = weights1[:,1:shape(weights1)[1]]
    regularized_penalty2 = weights2[:,1:shape(weights2)[1]]
    regularized_penalty1 = regularized_penalty1 ** 2
    regularized_penalty2 = regularized_penalty2 ** 2
    output_target_diff = (output - inputs)**2
    KL = sum(sparsityParam*log(sparsityParam/p_avg) + (1-sparsityParam)*log((1-sparsityParam)/(1-p_avg)))
    cost = sum(output_target_diff)/(2*nData) + 0.5 * lambda_val * (sum(regularized_penalty1) + sum(regularized_penalty2)) + beta*KL
    print 'Linear Decoder Cost: ', cost
    return cost

def costfunc_gpu_ReLU(x, *args):
    num_input,num_hidden,num_output,inputs,lambda_val,sparsityParam,beta = args
    num_weights1 = (num_input+1)*num_hidden
    x = gpu.garray(x)
    inputs = gpu.garray(inputs)
    #weights1 = gpu.garray(reshape(x[0:num_weights1],(num_hidden,num_input+1)))
    weights1 = x[0:num_weights1].reshape((num_hidden,num_input+1))
    #weights2 = gpu.garray(reshape(x[num_weights1:shape(x)[0]], (num_output,num_hidden+1)))
    weights2 = x[num_weights1:shape(x)[0]].reshape((num_output,num_hidden+1))
    nData = shape(inputs)[1]
    data = gpu.concatenate((gpu.ones((1,nData)), inputs), axis = 0)
    hidden_sum = gpu.dot(weights1, data)
    hidden_activation = gpu.log(1+hidden_sum.exp())
    p_avg = gpu.sum(hidden_activation,axis=1)/nData
    hidden_activation = gpu.concatenate((gpu.ones((1,nData)), hidden_activation), axis = 0)
    output = gpu.dot(weights2, hidden_activation)
    regularized_penalty1 = weights1[:,1:shape(weights1)[1]]
    regularized_penalty2 = weights2[:,1:shape(weights2)[1]]
    regularized_penalty1 = regularized_penalty1 * regularized_penalty1
    regularized_penalty2 = regularized_penalty2 * regularized_penalty2
    output_target_diff = (output - inputs)*(output - inputs)
    KL = gpu.sum(sparsityParam*gpu.log(sparsityParam/p_avg) + (1-sparsityParam)*gpu.log((1-sparsityParam)/(1-p_avg)))
    cost = gpu.sum(output_target_diff)/(2*nData) + 0.5 * lambda_val * (gpu.sum(regularized_penalty1) + gpu.sum(regularized_penalty2)) + beta*KL
    print 'ReLU Linear Decoder Cost: ', cost
    return cost

def costfunc_gpu(x, *args):
    num_input,num_hidden,num_output,inputs,lambda_val,sparsityParam,beta = args
    num_weights1 = (num_input+1)*num_hidden
    x = gpu.garray(x)
    inputs = gpu.garray(inputs)
    #weights1 = gpu.garray(reshape(x[0:num_weights1],(num_hidden,num_input+1)))
    weights1 = x[0:num_weights1].reshape((num_hidden,num_input+1))
    #weights2 = gpu.garray(reshape(x[num_weights1:shape(x)[0]], (num_output,num_hidden+1)))
    weights2 = x[num_weights1:shape(x)[0]].reshape((num_output,num_hidden+1))
    nData = shape(inputs)[1]
    data = gpu.concatenate((gpu.ones((1,nData)), inputs), axis = 0)
    hidden_sum = gpu.dot(weights1, data)
    hidden_activation = hidden_sum.logistic()
    p_avg = gpu.sum(hidden_activation,axis=1)/nData
    hidden_activation = gpu.concatenate((gpu.ones((1,nData)), hidden_activation), axis = 0)
    output = gpu.dot(weights2, hidden_activation)
    regularized_penalty1 = weights1[:,1:shape(weights1)[1]]
    regularized_penalty2 = weights2[:,1:shape(weights2)[1]]
    regularized_penalty1 = regularized_penalty1 * regularized_penalty1
    regularized_penalty2 = regularized_penalty2 * regularized_penalty2
    output_target_diff = (output - inputs)*(output - inputs)
    KL = gpu.sum(sparsityParam*gpu.log(sparsityParam/p_avg) + (1-sparsityParam)*gpu.log((1-sparsityParam)/(1-p_avg)))
    cost = gpu.sum(output_target_diff)/(2*nData) + 0.5 * lambda_val * (gpu.sum(regularized_penalty1) + gpu.sum(regularized_penalty2)) + beta*KL
    print 'Linear Decoder Cost: ', cost
    del x
    del inputs
    del data
    del hidden_sum
    del hidden_activation
    del p_avg
    del output
    del regularized_penalty1
    del regularized_penalty2
    del weights1
    del weights2
    del output_target_diff 
    gpu.free_reuse_cache()
    return cost

def costfunc_ReLU(x, *args):
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
    output = dot(weights2, hidden_activation)
    regularized_penalty1 = weights1[:,1:shape(weights1)[1]]
    regularized_penalty2 = weights2[:,1:shape(weights2)[1]]
    regularized_penalty1 = regularized_penalty1 ** 2
    regularized_penalty2 = regularized_penalty2 ** 2
    output_target_diff = (output - inputs)**2
    KL = sum(sparsityParam*log(sparsityParam/p_avg) + (1-sparsityParam)*log((1-sparsityParam)/(1-p_avg)))
    cost = sum(output_target_diff)/(2*nData) + 0.5 * lambda_val * (sum(regularized_penalty1) + sum(regularized_penalty2)) + beta*KL
    print 'Linear Decoder Cost: ', cost
    return cost

def grad_costfunc_gpu(x, *args):
    num_input,num_hidden,num_output,inputs,lambda_val,sparsityParam,beta = args
    num_weights1 = (num_input+1)*num_hidden
    num_weights2 = (num_hidden+1)*num_output
    x = gpu.garray(x)
    inputs = gpu.garray(inputs)
    weights1 = x[0:num_weights1].reshape((num_hidden,num_input+1))
    weights2 = x[num_weights1:shape(x)[0]].reshape((num_output,num_hidden+1))
    nData = shape(inputs)[1]
    data = gpu.concatenate((gpu.ones((1,nData)), inputs), axis = 0)
    hidden_sum = gpu.dot(weights1, data)
    hidden_activation = hidden_sum.logistic()
    p_avg = gpu.sum(hidden_activation,axis=1)/nData
    grad_sparse = -1*sparsityParam/p_avg.as_numpy_array() + (1-sparsityParam)/(1-p_avg.as_numpy_array())
    grad_sparse = append(0,grad_sparse)
    grad_sparse = tile(grad_sparse, (nData, 1))
    grad_sparse = gpu.garray(transpose(grad_sparse))
    hidden_activation = gpu.concatenate((gpu.ones((1,nData)), hidden_activation), axis = 0)
    outputs = gpu.dot(weights2, hidden_activation)
    weights1_grad = gpu.zeros(shape(weights1))
    weights2_grad = gpu.zeros(shape(weights2))
    p = outputs-inputs
    weights2_grad += gpu.dot(p, gpu.garray(transpose(hidden_activation.as_numpy_array())))
    q_temp = gpu.dot(gpu.garray(transpose(weights2.as_numpy_array())),p) + beta*grad_sparse
    #q = multiply(multiply(q_temp,hidden_activation),(1-hidden_activation))
    q = (q_temp*hidden_activation)*(1-hidden_activation)
    delta2 = gpu.dot(q, gpu.garray(transpose(data.as_numpy_array())))
    weights1_grad += delta2[1:shape(delta2)[0], :]
    weights1_grad = weights1_grad/nData
    weights2_grad = weights2_grad/nData
    weights1_grad[:,1:shape(weights1_grad)[1]] = weights1_grad[:,1:shape(weights1_grad)[1]] + weights1[:,1:shape(weights1)[1]] * lambda_val
    weights2_grad[:,1:shape(weights2_grad)[1]] = weights2_grad[:,1:shape(weights2_grad)[1]] + weights2[:,1:shape(weights2)[1]] * lambda_val
    #weights1_grad = reshape(weights1_grad, num_weights1)
    weights1_grad = weights1_grad.reshape(num_weights1)
    #weights2_grad = reshape(weights2_grad, num_weights2)
    weights2_grad = weights2_grad.reshape(num_weights2)
    del x
    del inputs
    del data
    del grad_sparse
    del p
    del q_temp
    del q
    del delta2
    del hidden_sum
    del hidden_activation
    del weights1
    del weights2
    gpu.free_reuse_cache()
    return hstack((weights1_grad.as_numpy_array(),weights2_grad.as_numpy_array()))

def grad_costfunc_gpu_ReLU(x, *args):
    num_input,num_hidden,num_output,inputs,lambda_val,sparsityParam,beta = args
    num_weights1 = (num_input+1)*num_hidden
    num_weights2 = (num_hidden+1)*num_output
    x = gpu.garray(x)
    inputs = gpu.garray(inputs)
    weights1 = x[0:num_weights1].reshape((num_hidden,num_input+1))
    weights2 = x[num_weights1:shape(x)[0]].reshape((num_output,num_hidden+1))
    nData = shape(inputs)[1]
    data = gpu.concatenate((gpu.ones((1,nData)), inputs), axis = 0)
    hidden_sum = gpu.dot(weights1, data)
    hidden_activation = gpu.log(1+hidden_sum.exp())
    p_avg = gpu.sum(hidden_activation,axis=1)/nData
    grad_sparse = -1*sparsityParam/p_avg.as_numpy_array() + (1-sparsityParam)/(1-p_avg.as_numpy_array())
    grad_sparse = append(0,grad_sparse)
    grad_sparse = tile(grad_sparse, (nData, 1))
    grad_sparse = gpu.garray(transpose(grad_sparse))
    hidden_activation = gpu.concatenate((gpu.ones((1,nData)), hidden_activation), axis = 0)
    outputs = gpu.dot(weights2, hidden_activation)
    weights1_grad = gpu.zeros(shape(weights1))
    weights2_grad = gpu.zeros(shape(weights2))
    p = outputs-inputs
    weights2_grad += gpu.dot(p, gpu.garray(transpose(hidden_activation.as_numpy_array())))
    q_temp = gpu.dot(gpu.garray(transpose(weights2.as_numpy_array())),p) + beta*grad_sparse
    #q = multiply(multiply(q_temp,hidden_activation),(1-hidden_activation))
    q = q_temp*hidden_sum.logistic()
    delta2 = gpu.dot(q, gpu.garray(transpose(data.as_numpy_array())))
    weights1_grad += delta2[1:shape(delta2)[0], :]
    weights1_grad = weights1_grad/nData
    weights2_grad = weights2_grad/nData
    weights1_grad[:,1:shape(weights1_grad)[1]] = weights1_grad[:,1:shape(weights1_grad)[1]] + weights1[:,1:shape(weights1)[1]] * lambda_val
    weights2_grad[:,1:shape(weights2_grad)[1]] = weights2_grad[:,1:shape(weights2_grad)[1]] + weights2[:,1:shape(weights2)[1]] * lambda_val
    #weights1_grad = reshape(weights1_grad, num_weights1)
    weights1_grad = weights1_grad.reshape(num_weights1)
    #weights2_grad = reshape(weights2_grad, num_weights2)
    weights2_grad = weights2_grad.reshape(num_weights2)
    return hstack((weights1_grad.as_numpy_array(),weights2_grad.as_numpy_array()))

def grad_costfunc_ReLU(x, *args):
    num_input,num_hidden,num_output,inputs,lambda_val,sparsityParam,beta = args
    num_weights1 = (num_input+1)*num_hidden
    num_weights2 = (num_hidden+1)*num_output
    weights1 = reshape(x[0:num_weights1],(num_hidden,num_input+1))
    weights2 = reshape(x[num_weights1:shape(x)[0]], (num_output,num_hidden+1))
    nData = shape(inputs)[1]
    x = concatenate((ones((1,nData)), inputs), axis = 0)
    hidden_sum = dot(weights1, x)
    hidden_activation = log(1 + exp(hidden_sum))
    hidden_derivative = 1/(1 + exp(-hidden_sum))
    p_avg = sum(hidden_activation,axis=1)/nData
    grad_sparse = -1*sparsityParam/p_avg + (1-sparsityParam)/(1-p_avg)
    grad_sparse = append(0,grad_sparse)
    grad_sparse = tile(grad_sparse, (nData, 1))
    grad_sparse = transpose(grad_sparse)
    hidden_activation = concatenate((ones((1,nData)), hidden_activation), axis = 0)
    hidden_derivative = concatenate((ones((1,nData)), hidden_derivative), axis = 0)
    outputs = dot(weights2, hidden_activation)
    weights1_grad = zeros(shape(weights1))
    weights2_grad = zeros(shape(weights2))
    p = outputs-inputs
    weights2_grad += dot(p, transpose(hidden_activation))
    q_temp = dot(transpose(weights2),p) + beta*grad_sparse
    q = multiply(q_temp,hidden_derivative)
    delta2 = dot(q, transpose(x))
    weights1_grad += delta2[1:shape(delta2)[0], :]
    weights1_grad = weights1_grad/nData
    weights2_grad = weights2_grad/nData
    weights1_grad[:,1:shape(weights1_grad)[1]] = weights1_grad[:,1:shape(weights1_grad)[1]] + weights1[:,1:shape(weights1)[1]] * lambda_val
    weights2_grad[:,1:shape(weights2_grad)[1]] = weights2_grad[:,1:shape(weights2_grad)[1]] + weights2[:,1:shape(weights2)[1]] * lambda_val
    weights1_grad = reshape(weights1_grad, num_weights1)
    weights2_grad = reshape(weights2_grad, num_weights2)
    return hstack((weights1_grad,weights2_grad))

def grad_costfunc(x, *args):
    num_input,num_hidden,num_output,inputs,lambda_val,sparsityParam,beta = args
    num_weights1 = (num_input+1)*num_hidden
    num_weights2 = (num_hidden+1)*num_output
    weights1 = reshape(x[0:num_weights1],(num_hidden,num_input+1))
    weights2 = reshape(x[num_weights1:shape(x)[0]], (num_output,num_hidden+1))
    nData = shape(inputs)[1]
    x = concatenate((ones((1,nData)), inputs), axis = 0)
    hidden_sum = dot(weights1, x)
    hidden_activation = 1 / (1+exp(-hidden_sum))
    p_avg = sum(hidden_activation,axis=1)/nData
    grad_sparse = -1*sparsityParam/p_avg + (1-sparsityParam)/(1-p_avg)
    grad_sparse = append(0,grad_sparse)
    grad_sparse = tile(grad_sparse, (nData, 1))
    grad_sparse = transpose(grad_sparse)
    hidden_activation = concatenate((ones((1,nData)), hidden_activation), axis = 0)
    outputs = dot(weights2, hidden_activation)
    weights1_grad = zeros(shape(weights1))
    weights2_grad = zeros(shape(weights2))
    p = outputs-inputs
    weights2_grad += dot(p, transpose(hidden_activation))
    q_temp = dot(transpose(weights2),p) + beta*grad_sparse
    #q = multiply(multiply(q_temp,hidden_activation),(1-hidden_activation))
    q = (q_temp*hidden_activation)*(1-hidden_activation)
    delta2 = dot(q, transpose(x))
    weights1_grad += delta2[1:shape(delta2)[0], :]
    weights1_grad = weights1_grad/nData
    weights2_grad = weights2_grad/nData
    weights1_grad[:,1:shape(weights1_grad)[1]] = weights1_grad[:,1:shape(weights1_grad)[1]] + weights1[:,1:shape(weights1)[1]] * lambda_val
    weights2_grad[:,1:shape(weights2_grad)[1]] = weights2_grad[:,1:shape(weights2_grad)[1]] + weights2[:,1:shape(weights2)[1]] * lambda_val
    weights1_grad = reshape(weights1_grad, num_weights1)
    weights2_grad = reshape(weights2_grad, num_weights2)
    return hstack((weights1_grad,weights2_grad))
    
def checkGradient():
    num_input = 8*8*3
    num_hidden = 10
    num_output = num_input
    lambda_val = 0.003
    sparsityParam = 0.035
    beta = 5
    data = scipy.io.loadmat('stlSampledPatches.mat')
    patches = data['patches']
    inputs = patches[:,0:10]
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
    print numgrad
    print grad
    diff = linalg.norm(numgrad-grad)/linalg.norm(numgrad+grad)
    return diff

def checkGradientGPU():
    num_input = 8*8*3
    num_hidden = 10
    num_output = num_input
    lambda_val = 0.003
    sparsityParam = 0.035
    beta = 5
    data = scipy.io.loadmat('stlSampledPatches.mat')
    patches = data['patches']
    inputs = patches[:,0:10]
    r = gpu.sqrt(6)/gpu.sqrt(num_hidden+num_input+1)
    weights1 = (gpu.rand(num_hidden,num_input+1))*2*r-r
    weights2 = (gpu.rand(num_output,num_hidden+1))*2*r-r
    num_weights1 = (num_input+1)*num_hidden
    num_weights2 = (num_hidden+1)*num_output
    weights1 = weights1.reshape(num_weights1)
    weights2 = weights2.reshape(num_weights2)
    weights = hstack((weights1.as_numpy_array(),weights2.as_numpy_array()))
    args = (num_input,num_hidden,num_output,inputs,lambda_val,sparsityParam,beta)
    numgrad = zeros(size(weights))
    numgrad2 = zeros(size(weights))
    perturb = zeros(size(weights))
    e = 1e-4
    for p in range(size(weights)):
        perturb[p] = e;
        minus_weights = weights - perturb
        plus_weights = weights + perturb
        loss1 = costfunc_gpuTRY(minus_weights, *args)
	lossc1 = costfunc(minus_weights, *args)
        loss2 = costfunc_gpuTRY(plus_weights, *args)
	lossc2 = costfunc(plus_weights, *args)
        numgrad[p] = (loss2 - loss1) / (2*e)
	numgrad2[p] = (lossc2 - lossc1) / (2*e)
        perturb[p] = 0
    grad = grad_costfunc_gpu(weights, *args)
    grad2 = grad_costfunc(weights, *args)
    diff = linalg.norm(numgrad-grad)/linalg.norm(numgrad+grad)
    diff2 = linalg.norm(numgrad2-grad2)/linalg.norm(numgrad2+grad2)
    diff3 = linalg.norm(numgrad-grad2)/linalg.norm(numgrad+grad2)
    diff4 = linalg.norm(numgrad2-grad)/linalg.norm(numgrad2+grad)
    diffnum = linalg.norm(numgrad2-numgrad)/linalg.norm(numgrad2+numgrad)
    diffgrad = linalg.norm(grad2-grad)/linalg.norm(grad2+grad)
    print "pure GPU difference:",diff
    print "pure CPU difference:",diff2
    print "GPU cost, CPU grad:",diff3
    print "CPU cost, GPU grad:",diff4
    print "CPU cost and GPU cost difference:",diffnum
    print "CPU grad and GPU grad difference:",diffgrad
    return "OK"

def linear_decoder_run_gpu(data, numInput, numHidden):
    print "Starting Feature Abstraction..."
    num_input = numInput
    num_hidden = numHidden
    num_output = numInput
    lambda_val = 3e-3
    sparsityParam = 0.035
    beta = 5
    inputs = data
    r = gpu.sqrt(6)/gpu.sqrt(num_hidden+num_input+1)
    weights1 = (gpu.rand(num_hidden,num_input+1))*2*r-r
    weights2 = (gpu.rand(num_output,num_hidden+1))*2*r-r
    num_weights1 = (num_input+1)*num_hidden
    num_weights2 = (num_hidden+1)*num_output
    #weights1 = reshape(weights1, num_weights1)
    weights1 = weights1.reshape(num_weights1)
    #weights2 = reshape(weights2, num_weights2)
    weights2 = weights2.reshape(num_weights2)
    weights = hstack((weights1.as_numpy_array(),weights2.as_numpy_array()))
    args = (num_input,num_hidden,num_output,inputs,lambda_val,sparsityParam,beta)
    opttheta = optimize.fmin_l_bfgs_b(costfunc_gpu, weights, fprime=grad_costfunc_gpu, args=args, maxiter=400)
    weights = opttheta[0]
    weights1 = reshape(weights[0:num_weights1],(num_hidden,num_input+1))
    weights2 = reshape(weights[num_weights1:shape(weights)[0]], (num_output,num_hidden+1))
    scipy.io.savemat('learntFeaturesGPU.mat', mdict={'learntFeatures': weights1})
    return weights1
    
def linear_decoder_run_ReLU(data, numInput, numHidden):
    print "Starting Feature Abstraction..."
    num_input = numInput
    num_hidden = numHidden
    num_output = numInput
    lambda_val = 3e-3
    sparsityParam = 0.035
    beta = 5
    inputs = data
    r = sqrt(6)/sqrt(num_hidden+num_input+1)
    weights1 = (random.rand(num_hidden,num_input+1))*2*r-r
    weights2 = (random.rand(num_output,num_hidden+1))*2*r-r
    num_weights1 = (num_input+1)*num_hidden
    num_weights2 = (num_hidden+1)*num_output
    weights1 = reshape(weights1, num_weights1)
    weights2 = reshape(weights2, num_weights2)
    weights = hstack((weights1,weights2))
    args = (num_input,num_hidden,num_output,inputs,lambda_val,sparsityParam,beta)
    opttheta = optimize.fmin_l_bfgs_b(costfunc_ReLU, weights, fprime=grad_costfunc_ReLU, args=args, maxiter=300)
    weights = opttheta[0]
    weights1 = reshape(weights[0:num_weights1],(num_hidden,num_input+1))
    weights2 = reshape(weights[num_weights1:shape(weights)[0]], (num_output,num_hidden+1))
    scipy.io.savemat('learntFeaturesReLU.mat', mdict={'learntFeatures': weights1})
    return weights1

def linear_decoder_run(data, numInput, numHidden):
    print "Starting Feature Abstraction..."
    num_input = numInput
    num_hidden = numHidden
    num_output = numInput
    lambda_val = 3e-3
    sparsityParam = 0.035
    beta = 5
    inputs = data
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
    scipy.io.savemat('learntFeatures.mat', mdict={'learntFeatures': weights1})
    return weights1

def test_run():
    data = scipy.io.loadmat('stlSampledPatches.mat')
    patches = data['patches']
    numPatches = shape(patches)[1]
    meanPatch = mean(patches, axis = 1)
    patches = transpose(patches)-meanPatch
    patches = transpose(patches)
    sigma = dot(patches, transpose(patches)) / numPatches
    U,S,V = linalg.svd(sigma)
    ZCAWhiteTemp = dot(U, diag(1/sqrt(S + 0.1)))
    ZCAWhite = dot(ZCAWhiteTemp, transpose(U))
    patches = dot(ZCAWhite, patches)
    scipy.io.savemat('patches_zca.mat', mdict={'patches_zca': patches})
    patchDim = 8
    imageChannel = 3
    numInput = patchDim * patchDim * imageChannel
    numHidden = 400
    result = linear_decoder_run_gpu(patches, numInput, numHidden)
    return result

def timeIt():
    start = time.clock()
    test_run()
    end = time.clock()
    print "Total running time:", end-start
