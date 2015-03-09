# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 21:21:01 2014

@author: himwaing
"""

from numpy import *
from scipy import optimize
from linearDecoderReLU import *
from optimized_mlp_gpu import *
from mlp_softmax_gpu1layerReLU import *
from mlp_softmax_gpuReLU import *
from softmax import *
import scipy.io
import gnumpy as gpu

def multilayer_feature_learning(data, inputSize, l1Size, l2Size, sparsityParam, lambda_val, beta):
    print "Now starting feature abstraction..."
    num_input = inputSize
    num_hidden_L1 = l1Size
    num_hidden_L2 = l2Size
    num_output_L1 = inputSize
    num_output_L2 = num_hidden_L1
    sparsityParam = sparsityParam
    lambda_val = lambda_val
    beta = beta
    inputs = gpu.garray(data)
    r = gpu.sqrt(6)/gpu.sqrt(num_hidden_L1+num_input+1)
    weights1_L1 = (gpu.rand(num_hidden_L1,num_input+1))*2*r-r
    weights2_L1 = (gpu.rand(num_output_L1,num_hidden_L1+1))*2*r-r
    num_weights1_L1 = (num_input+1)*num_hidden_L1
    num_weights2_L1 = (num_hidden_L1+1)*num_output_L1
    weights1_L1 = weights1_L1.reshape(num_weights1_L1)
    weights2_L1 = weights2_L1.reshape(num_weights2_L1)
    weights_L1 = hstack((weights1_L1.as_numpy_array(),weights2_L1.as_numpy_array()))
    print "Level 1 Abstraction Starting...."
    weights_L1 = linear_decoder_run_ReLU(data, weights_L1, num_input, num_hidden_L1)
    weights1_L1 = weights_L1[0:num_weights1_L1].reshape((num_hidden_L1,num_input+1))
    weights2_L1 = weights_L1[num_weights1_L1:shape(weights_L1)[0]].reshape((num_output_L1,num_hidden_L1+1))
    scipy.io.savemat('HiggsBosonLevel1.mat', mdict={'learntFeaturesL1_1': weights1_L1, 'learntFeaturesL1_2': weights2_L1})
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
    weights1_L2 = weights1_L2.reshape(num_weights1_L2)
    weights2_L2 = weights2_L2.reshape(num_weights2_L2)
    weights_L2 = hstack((weights1_L2.as_numpy_array(),weights2_L2.as_numpy_array()))
    print "Level 2 Abstraction Starting...."
    weights_L2 = linear_decoder_run_ReLU(L1_activation, weights_L2, num_hidden_L1, num_hidden_L2)
    weights1_L2 = weights_L2[0:num_weights1_L2].reshape((num_hidden_L2,num_hidden_L1+1))
    weights2_L2 = weights_L2[num_weights1_L2:shape(weights_L2)[0]].reshape((num_output_L2,num_hidden_L2+1))
    scipy.io.savemat('HiggsBosonLevel2.mat', mdict={'learntFeaturesL2_1': weights1_L2,'learntFeaturesL2_2': weights2_L2})
    L2_activation = feedforward(weights1_L2, L1_activation)
    del weights_L2
    del weights1_L2
    del weights2_L2
    gpu.free_reuse_cache()
    gpu.free_reuse_cache()
    print "Abstraction completed."
    return L2_activation

def feedforward(theta, data):
    nData = shape(data)[1]
    x = gpu.concatenate((gpu.ones((1,nData)), data), axis = 0)
    hidden_sum = gpu.dot(theta, x)
    relu_mask_hidden = gpu.ones(shape(hidden_sum)) * (hidden_sum>0)
    hidden_activation = hidden_sum*relu_mask_hidden
    return hidden_activation

def fine_tuning_cost_gpu(x, *args):
    inputSize, l1Size, l2Size, l3Size, lambda_val, inputs = args
    num_weights_L1 = l1Size * (inputSize + 1)
    num_weights_L2 = l2Size * (l1Size + 1)
    num_weights_L3 = l3Size * (l2Size + 1)
    x = gpu.garray(x)
    inputs = gpu.garray(inputs)
    weights1 = x[0:num_weights_L1].reshape((l1Size, inputSize + 1))
    weights2 = x[num_weights_L1:num_weights_L1+num_weights_L2].reshape((l2Size, l1Size + 1))
    weights3 = x[num_weights_L1+num_weights_L2:num_weights_L1+num_weights_L2+num_weights_L3].reshape((l3Size, l2Size + 1))
    weights4 = x[num_weights_L1+num_weights_L2+num_weights_L3:shape(x)[0]].reshape((inputSize, l3Size + 1))
    nData = shape(inputs)[1]
    x = gpu.concatenate((gpu.ones((1,nData)), inputs), axis = 0)
    hidden1_sum = gpu.dot(weights1, x)
    #hidden1_activation = gpu.log(1+hidden1_sum.exp())
    relu_mask_hidden1 = gpu.ones(shape(hidden1_sum)) * (hidden1_sum>0)
    hidden1_activation = hidden1_sum*relu_mask_hidden1
    hidden1_activation = gpu.concatenate((gpu.ones((1,nData)), hidden1_activation), axis = 0)
    hidden2_sum = gpu.dot(weights2, hidden1_activation)
    #hidden2_activation = gpu.log(1+hidden2_sum.exp())
    relu_mask_hidden2 = gpu.ones(shape(hidden2_sum)) * (hidden2_sum>0)
    hidden2_activation = hidden2_sum*relu_mask_hidden2
    hidden2_activation = gpu.concatenate((gpu.ones((1,nData)), hidden2_activation), axis = 0)
    hidden3_sum = gpu.dot(weights3, hidden2_activation)
    hidden3_activation = hidden3_sum
    hidden3_activation = gpu.concatenate((gpu.ones((1,nData)), hidden3_activation), axis = 0)
    output_sum = gpu.dot(weights4, hidden3_activation)
    outputs = output_sum
    regularized_penalty3 = weights3[:,1:shape(weights3)[1]]
    regularized_penalty4 = weights4[:,1:shape(weights4)[1]]
    regularized_penalty3 = regularized_penalty3 ** 2
    regularized_penalty4 = regularized_penalty4 ** 2
    output_target_diff = (outputs - inputs)**2
    cost = gpu.sum(output_target_diff)/(2*nData) + 0.5 * lambda_val * (gpu.sum(regularized_penalty3) + gpu.sum(regularized_penalty4))
    print 'Fine Tuning Cost: ', cost
    return cost

def fine_tuning_grad_gpu(x, *args):
    inputSize, l1Size, l2Size, l3Size, lambda_val, inputs = args
    num_weights_L1 = l1Size * (inputSize + 1)
    num_weights_L2 = l2Size * (l1Size + 1)
    num_weights_L3 = l3Size * (l2Size + 1)
    num_weights_L4 = inputSize * (l3Size + 1)
    x = gpu.garray(x)
    inputs = gpu.garray(inputs)
    weights1 = x[0:num_weights_L1].reshape((l1Size, inputSize + 1))
    weights2 = x[num_weights_L1:num_weights_L1+num_weights_L2].reshape((l2Size, l1Size + 1))
    weights3 = x[num_weights_L1+num_weights_L2:num_weights_L1+num_weights_L2+num_weights_L3].reshape((l3Size, l2Size + 1))
    weights4 = x[num_weights_L1+num_weights_L2+num_weights_L3:shape(x)[0]].reshape((inputSize, l3Size + 1))
    nData = shape(inputs)[1]
    x = gpu.concatenate((gpu.ones((1,nData)), inputs), axis = 0)
    hidden1_sum = gpu.dot(weights1, x)
    #hidden1_activation = gpu.log(1+hidden1_sum.exp())
    #hidden1_derivative = hidden1_sum.logistic()
    relu_mask_hidden1 = gpu.ones(shape(hidden1_sum)) * (hidden1_sum>0)
    hidden1_activation = hidden1_sum*relu_mask_hidden1
    hidden1_derivative = relu_mask_hidden1
    hidden1_activation = gpu.concatenate((gpu.ones((1,nData)), hidden1_activation), axis = 0)
    hidden1_derivative = gpu.concatenate((gpu.ones((1,nData)), hidden1_derivative), axis = 0)
    hidden2_sum = gpu.dot(weights2, hidden1_activation)
    #hidden2_activation = gpu.log(1+hidden2_sum.exp())
    #hidden2_derivative = hidden2_sum.logistic()
    relu_mask_hidden2 = gpu.ones(shape(hidden2_sum)) * (hidden2_sum>0)
    hidden2_activation = hidden2_sum*relu_mask_hidden2
    hidden2_derivative = relu_mask_hidden2
    hidden2_activation = gpu.concatenate((gpu.ones((1,nData)), hidden2_activation), axis = 0)
    hidden2_derivative = gpu.concatenate((gpu.ones((1,nData)), hidden2_derivative), axis = 0)
    hidden3_sum = gpu.dot(weights3, hidden2_activation)
    hidden3_activation = hidden3_sum
    hidden3_activation = gpu.concatenate((gpu.ones((1,nData)), hidden3_activation), axis = 0)
    output_sum = gpu.dot(weights4, hidden3_activation)
    outputs = output_sum
    weights1_grad = gpu.zeros(shape(weights1))
    weights2_grad = gpu.zeros(shape(weights2))
    weights3_grad = gpu.zeros(shape(weights3))
    weights4_grad = gpu.zeros(shape(weights4))
    a = (outputs - inputs) 
    weights4_grad += gpu.dot(a, gpu.garray(transpose(hidden3_activation.as_numpy_array())))
    b_temp = gpu.dot(gpu.garray(transpose(weights4.as_numpy_array())),a)
    b = b_temp
    delta2 = gpu.dot(b, gpu.garray(transpose(hidden2_activation.as_numpy_array())))
    weights3_grad += delta2[1:shape(delta2)[0], :]
    c_temp = gpu.dot(gpu.garray(transpose(weights3.as_numpy_array())), b[1:shape(b)[0], :])
    c = c_temp*hidden2_derivative
    delta3 = gpu.dot(c, gpu.garray(transpose(hidden1_activation.as_numpy_array())))
    weights2_grad += delta3[1:shape(delta3)[0], :]
    d_temp = gpu.dot(gpu.garray(transpose(weights2.as_numpy_array())), c[1:shape(c)[0], :])
    d = d_temp*hidden1_derivative
    delta4 = gpu.dot(d, gpu.garray(transpose(x.as_numpy_array())))
    weights1_grad += delta4[1:shape(delta4)[0], :]
    weights1_grad = weights1_grad/nData
    weights2_grad = weights2_grad/nData
    weights3_grad = weights3_grad/nData
    weights4_grad = weights4_grad/nData
    weights3_grad[:,1:shape(weights3_grad)[1]] = weights3_grad[:,1:shape(weights3_grad)[1]] + weights3[:,1:shape(weights3)[1]] * lambda_val
    weights4_grad[:,1:shape(weights4_grad)[1]] = weights4_grad[:,1:shape(weights4_grad)[1]] + weights4[:,1:shape(weights4)[1]] * lambda_val
    weights1_grad = reshape(weights1_grad.as_numpy_array(), num_weights_L1)
    weights2_grad = reshape(weights2_grad.as_numpy_array(), num_weights_L2)
    weights3_grad = reshape(weights3_grad.as_numpy_array(), num_weights_L3)
    weights4_grad = reshape(weights4_grad.as_numpy_array(), num_weights_L4)
    return hstack((weights1_grad,weights2_grad,weights3_grad,weights4_grad))

def running(inputData, l1Size, l2Size):
    inputs = inputData
    inputSize = 30
    sparsityParam = 0.05
    lambda_val = 7e-5
    lambda_valFineTune = 1e-5
    beta = 3
    multilayer_feature_learning(inputs, inputSize, l1Size, l2Size, sparsityParam, lambda_val, beta)
    weights1 = scipy.io.loadmat('HiggsBosonLevel1.mat')['learntFeaturesL1_1']
    weights2 = scipy.io.loadmat('HiggsBosonLevel2.mat')['learntFeaturesL2_1']
    weights3 = scipy.io.loadmat('HiggsBosonLevel2.mat')['learntFeaturesL2_2']
    weights4 = scipy.io.loadmat('HiggsBosonLevel1.mat')['learntFeaturesL1_2']
    gpu.free_reuse_cache()
    print "Fine Tuning the abstraction network..."
    num_input = inputSize
    num_hidden1 = l1Size
    num_hidden2 = l2Size
    num_hidden3 = l1Size
    num_output = num_input
    num_weights1 = (num_input+1)*num_hidden1
    num_weights2 = (num_hidden1+1)*num_hidden2
    num_weights3 = (num_hidden2+1)*num_hidden3
    num_weights4 = (num_hidden3+1)*num_output
    weights1 = reshape(weights1, num_weights1)
    weights2 = reshape(weights2, num_weights2)
    weights3 = reshape(weights3, num_weights3)
    weights4 = reshape(weights4, num_weights4)
    weights = hstack((weights1,weights2,weights3,weights4))
    print "Fine Tuning Starting..."
    stepSize = 200000
    for i in range(int(shape(inputs)[1]/stepSize)):
        print "Batch:", i
        data = inputs[:,i*stepSize:(i+1)*stepSize]
        args = (num_input, num_hidden1, num_hidden2, num_hidden3, lambda_valFineTune, data)
        opttheta = optimize.fmin_l_bfgs_b(fine_tuning_cost_gpu, weights, fprime=fine_tuning_grad_gpu, args=args, maxiter=200)
        weights = opttheta[0]
        del opttheta
        gpu.free_reuse_cache()
    weights1 = reshape(weights[0:num_weights1], (l1Size, inputSize + 1))
    weights2 = reshape(weights[num_weights1:num_weights1+num_weights2], (l2Size, l1Size + 1))
    weights3 = reshape(weights[num_weights1+num_weights2:num_weights1+num_weights2+num_weights3], (num_hidden3, l2Size + 1))
    weights4 = reshape(weights[num_weights1+num_weights2+num_weights3:shape(weights)[0]], (inputSize, num_hidden3 + 1))
    scipy.io.savemat('HiggsBoson_FineTuned_features2Layers.mat', mdict={'learntFeaturesL1': weights1,'learntFeaturesL2': weights2, 'learntFeaturesL3': weights3, 'learntFeaturesL4': weights4})
    return weights1, weights2

def train(trainData,numTrain,adaBoostWeights, networkNum):
    label  = trainData[:,32]
    oppoLabel = 1 - label
    groundTruth = array([oppoLabel,label])
    #groundTruth = array([label,oppoLabel])
    trainGT = groundTruth[:,0:numTrain]
    #trainGT = groundTruth
    all_data   = trainData[:,1:31]
    all_data = transpose(all_data)
    #all_data_mean = all_data.mean(axis=0)
    #all_data_std = all_data.std(axis=0)
    #all_data = (all_data - all_data_mean)/all_data_std
    #all_data_max = abs(all_data).max(axis=0)
    #all_data = all_data/all_data_max
    all_data = all_data/999
    data = all_data[:,0:numTrain]
    originalData = data
    data = data * adaBoostWeights
    print data
    #combinedData = append(data,trainGT, axis=0)
    #data = all_data
    actualTestData = all_data[:,numTrain:shape(all_data)[1]]    
    testData = originalData
    gpu.free_reuse_cache()
    l1Size = 150
    l2Size = 60
    #weights1, weights2 = running(data,l1Size, l2Size)
    weights1 = scipy.io.loadmat('HiggsBoson_FineTuned_features2Layers.mat')['learntFeaturesL1']
    weights2 = scipy.io.loadmat('HiggsBoson_FineTuned_features2Layers.mat')['learntFeaturesL2']
    nData = shape(data)[1]
    inputSize = shape(data)[0]
    x = concatenate((ones((1,nData)), data), axis = 0)
    hidden1_sum = dot(weights1, x)
    relu_mask_hidden1 = ones(shape(hidden1_sum)) * (hidden1_sum>0)
    hidden1_activation = hidden1_sum*relu_mask_hidden1
    hidden1_activation = concatenate((ones((1,nData)), hidden1_activation), axis = 0)
    hidden2_sum = dot(weights2, hidden1_activation)
    relu_mask_hidden2 = ones(shape(hidden2_sum)) * (hidden2_sum>0)
    hidden2_activation = hidden2_sum*relu_mask_hidden2
    scipy.io.savemat('hidden_activation_twoLayers.mat', mdict={'layer2Active':hidden2_activation})
    dataSoftmax = hidden2_activation
    inputSizeSoftmax = shape(dataSoftmax)[0]
    numClasses = shape(groundTruth)[0]
    l3Size = 20
    lambda_val = 4e-4
    lambda_val2 = 5e-6
    r = sqrt(6)/sqrt(numClasses+l3Size+inputSizeSoftmax)
    theta_hidden = (random.rand(l3Size,inputSizeSoftmax+1))*2*r-r
    theta_softmax = (random.rand(numClasses,l3Size))*2*r-r
    theta_hidden = reshape(theta_hidden, l3Size * (inputSizeSoftmax+1))
    theta_softmax = reshape(theta_softmax, numClasses * l3Size)
    theta = hstack((theta_hidden, theta_softmax))
    args = (numClasses, inputSizeSoftmax, l3Size, lambda_val, lambda_val2, dataSoftmax, gpu.garray(trainGT))
    print ("Now Initializing the Softmax layer...")
    opttheta = optimize.fmin_l_bfgs_b(mlpSoftmax1Layer_costfunc, theta, fprime=mlpSoftmax1Layer_grad, args=args, maxiter=300)
    theta = opttheta[0]
    num_weights_hidden = l3Size * (inputSizeSoftmax+1)
    theta_hidden = reshape(theta[0:num_weights_hidden], (l3Size, inputSizeSoftmax+1))
    theta_softmax = reshape(theta[num_weights_hidden:shape(theta)[0]], (numClasses, l3Size))
    predictionAccuracy(weights1, weights2, theta_hidden, theta_softmax, testData, label[0:numTrain])
    #predictionAccuracy(weights1, weights2, theta_hidden, theta_softmax, testData, label)
    print ("Final Stage: Deep Learning")
    num_weights1 = (inputSize+1)*l1Size
    num_weights2 = (l1Size+1)*l2Size
    num_theta = numClasses * l3Size
    weights1 = reshape(weights1, num_weights1)
    weights2 = reshape(weights2, num_weights2)
    theta_hidden = reshape(theta_hidden, num_weights_hidden)
    theta_softmax = reshape(theta_softmax, num_theta)
    weights = hstack((weights1,weights2,theta_hidden,theta_softmax))
    lambda_softmax = 4e-4
    lambda_hidden = 6e-6
    dropout_probability = 1
    stepSize = 50000
    #random.shuffle(combinedData.T)
    #data = combinedData[0:30,:]
    #trainGT = combinedData[30:32,:]
    for i in range(int(nData/stepSize)):
        print "Fine Tuning Batch:",i
        inputs = data[:,i*stepSize:(i+1)*stepSize]
        traingt = trainGT[:,i*stepSize:(i+1)*stepSize]
        args = (numClasses, inputSize, l1Size, l2Size, l3Size, lambda_softmax, lambda_hidden, inputs, squeeze(label), gpu.garray(traingt),dropout_probability)
        opttheta = optimize.fmin_l_bfgs_b(mlpSoftmax_costfunc, weights, args=args, maxiter=1000)
        weights = opttheta[0]
#    args = (numClasses, inputSize, l1Size, l2Size, l3Size, lambda_softmax, lambda_hidden, data, label[0:200000], gpu.garray(trainGT),dropout_probability)
#    opttheta = optimize.fmin_l_bfgs_b(mlpSoftmax_costfunc, weights, args=args, maxiter=15000)
#    weights = opttheta[0]
    print("Training Completed!")
    #weights1 = reshape(weights[0:num_weights1], (l1Size, inputSize + 1))
    weights1 = reshape(weights[0:num_weights1], (l1Size, inputSize + 1)) * dropout_probability
    #weights2 = reshape(weights[num_weights1:num_weights1+num_weights2], (l2Size, l1Size + 1))
    weights2 = reshape(weights[num_weights1:num_weights1+num_weights2], (l2Size, l1Size + 1)) * dropout_probability
    #theta_hidden = reshape(weights[num_weights1+num_weights2:num_weights1+num_weights2+num_weights_hidden], (l3Size, l2Size+1))
    theta_hidden = reshape(weights[num_weights1+num_weights2:num_weights1+num_weights2+num_weights_hidden], (l3Size, l2Size+1)) * dropout_probability
    theta_softmax = reshape(weights[num_weights1+num_weights2+num_weights_hidden:shape(weights)[0]], (numClasses, l3Size))
    numCasesPred = shape(testData)[1]
    numCasesActualPred = shape(actualTestData)[1]
    testData = concatenate((ones((1,numCasesPred)), testData), axis = 0)
    hidden_sum_L1 = dot(weights1, testData)
    relu_mask_hidden1 = ones(shape(hidden_sum_L1)) * (hidden_sum_L1>0)
    hidden_activation_L1 = hidden_sum_L1*relu_mask_hidden1
    hidden_activation_L1 = concatenate((ones((1,numCasesPred)), hidden_activation_L1), axis=0)
    hidden_sum_L2 = dot(weights2, hidden_activation_L1)
    relu_mask_hidden2 = ones(shape(hidden_sum_L2)) * (hidden_sum_L2>0)
    hidden_activation_L2 = hidden_sum_L2*relu_mask_hidden2
    hidden_activation_L2 = concatenate((ones((1,numCasesPred)), hidden_activation_L2), axis=0)
    hidden_sum_L3 = dot(theta_hidden, hidden_activation_L2)
    relu_mask_hidden3 = ones(shape(hidden_sum_L3)) * (hidden_sum_L3>0)
    hidden_activation_L3 = hidden_sum_L3*relu_mask_hidden3
    hidden_sum_softmax = dot(theta_softmax, hidden_activation_L3)
    hidden_sum_softmax = hidden_sum_softmax - hidden_sum_softmax.max(axis = 0)
    predictions = exp(hidden_sum_softmax)
    predictions = predictions / predictions.sum(axis = 0)
    pred = predictions.argmax(axis=0)
    accuracy = mean(pred == label[0:numTrain]) * 100
    #accuracy = mean(pred == label) * 100
    print "Original data Training Accuracy: ", accuracy, "%"
    actualTestData = concatenate((ones((1,numCasesActualPred)), actualTestData), axis = 0)
    hidden_sum_L1 = dot(weights1, actualTestData)
    relu_mask_hidden1 = ones(shape(hidden_sum_L1)) * (hidden_sum_L1>0)
    hidden_activation_L1 = hidden_sum_L1*relu_mask_hidden1
    hidden_activation_L1 = concatenate((ones((1,numCasesActualPred)), hidden_activation_L1), axis=0)
    hidden_sum_L2 = dot(weights2, hidden_activation_L1)
    relu_mask_hidden2 = ones(shape(hidden_sum_L2)) * (hidden_sum_L2>0)
    hidden_activation_L2 = hidden_sum_L2*relu_mask_hidden2
    hidden_activation_L2 = concatenate((ones((1,numCasesActualPred)), hidden_activation_L2), axis=0)
    hidden_sum_L3 = dot(theta_hidden, hidden_activation_L2)
    relu_mask_hidden3 = ones(shape(hidden_sum_L3)) * (hidden_sum_L3>0)
    hidden_activation_L3 = hidden_sum_L3*relu_mask_hidden3
    hidden_sum_softmax = dot(theta_softmax, hidden_activation_L3)
    hidden_sum_softmax = hidden_sum_softmax - hidden_sum_softmax.max(axis = 0)
    predictions = exp(hidden_sum_softmax)
    predictions = predictions / predictions.sum(axis = 0)
    predTest = predictions.argmax(axis=0)
    accuracy = mean(predTest == label[numTrain:shape(all_data)[1]]) * 100
    #accuracy = mean(pred == label) * 100
    print "Testing Accuracy: ", accuracy, "%"
    scipy.io.savemat('HiggsBoson_adaBoost' + str(networkNum) + '.mat', mdict={'learntFeaturesL1': weights1,'learntFeaturesL2': weights2, 'learntFeaturesL3': theta_hidden, 'learntFeaturesL4': theta_softmax, 'accuracy': accuracy})
    return pred, predictions, weights1, weights2, theta_hidden, theta_softmax
    
def adaBoost(numNetworks, numTrain):
    trainData = loadtxt('training.csv', delimiter=',', skiprows=1, converters={32: lambda x:int(x=='s'.encode('utf-8'))}) 
    print ('finish loading from csv ')
    ids = trainData[:,0]
    ids = array([ids])
    #trainIDs = ids[:,0:200000]
    testIDs = ids[:,numTrain:shape(ids)[1]]
    label  = trainData[:,32]
    trainLabel = label[0:numTrain]
    adaBoostWeights = ones((30, numTrain)) * 1.0
    alphas = zeros(numNetworks)
    sumOutput = zeros((2,50000))
    all_data   = trainData[:,1:31]
    all_data = transpose(all_data)/999
    for i in range(numNetworks):
        prediction, testProbability, weights1, weights2, theta_hidden, theta_softmax = train(trainData,numTrain,adaBoostWeights,i)
        errors = 1.0 * (prediction != trainLabel)
        errorRatio = sum(errors)/len(errors)
        alpha = 0.5 * log((1-errorRatio)/errorRatio)
        alphas[i] = alpha
        for j in range(len(errors)):
            if (errors[j] == 0):
                errors[j] = float(exp(-alpha))
            else: errors[j] = float(exp(alpha))
        print errors
        adaBoostWeights = (adaBoostWeights * errors)
        #adaBoostWeights = adaBoostWeights/sum(adaBoostWeights)
        sumOutput = sumOutput + alpha * testProbability
    sumOutput = sumOutput / sumOutput.sum(axis = 0)
    pred = sumOutput.argmax(axis=0)
    accuracy = mean(pred == label[numTrain:shape(all_data)[1]]) * 100
    print "Testing accuracy after adaBoost: ", accuracy, "%"
    sumOutput = concatenate((testIDs,sumOutput))
    pred = array([pred])
    labels = array([label[numTrain:shape(all_data)[1]]])
    pred = concatenate((testIDs,pred))
    labels = concatenate((testIDs,labels))
    scipy.io.savemat("predicting2and1layers_adaBoost.mat", mdict={'probability':sumOutput, 'prediction':pred,'labels':labels, 'alphas':alphas})
    return alphas

def modelTest(model, trainData):
    numTrain = 200000
    all_data   = trainData[:,1:31]
    all_data = transpose(all_data)
    actualTestData = all_data[:,0:numTrain]/999
    label  = trainData[:,32]
    numCasesActualPred = shape(actualTestData)[1]
    model = scipy.io.loadmat('HiggsBoson_adaBoost' + str(model) + '.mat')
    weights1 = model['learntFeaturesL1']
    weights2 = model['learntFeaturesL2']
    theta_hidden = model['learntFeaturesL3']
    theta_softmax = model['learntFeaturesL4']
    actualTestData = concatenate((ones((1,numCasesActualPred)), actualTestData), axis = 0)
    hidden_sum_L1 = dot(weights1, actualTestData)
    relu_mask_hidden1 = ones(shape(hidden_sum_L1)) * (hidden_sum_L1>0)
    hidden_activation_L1 = hidden_sum_L1*relu_mask_hidden1
    hidden_activation_L1 = concatenate((ones((1,numCasesActualPred)), hidden_activation_L1), axis=0)
    hidden_sum_L2 = dot(weights2, hidden_activation_L1)
    relu_mask_hidden2 = ones(shape(hidden_sum_L2)) * (hidden_sum_L2>0)
    hidden_activation_L2 = hidden_sum_L2*relu_mask_hidden2
    hidden_activation_L2 = concatenate((ones((1,numCasesActualPred)), hidden_activation_L2), axis=0)
    hidden_sum_L3 = dot(theta_hidden, hidden_activation_L2)
    relu_mask_hidden3 = ones(shape(hidden_sum_L3)) * (hidden_sum_L3>0)
    hidden_activation_L3 = hidden_sum_L3*relu_mask_hidden3
    hidden_sum_softmax = dot(theta_softmax, hidden_activation_L3)
    hidden_sum_softmax = hidden_sum_softmax - hidden_sum_softmax.max(axis = 0)
    predictions = exp(hidden_sum_softmax)
    predictions = predictions / predictions.sum(axis = 0)
    print predictions
    predTest = predictions.argmax(axis=0)
    accuracy = mean(predTest == label[0:numTrain]) * 100
    print "Testing Accuracy: ", accuracy, "%"
    return 
    
def runModels():
    trainData = loadtxt('training.csv', delimiter=',', skiprows=1, converters={32: lambda x:int(x=='s'.encode('utf-8'))}) 
    for i in range(3):
        modelTest(i,trainData)

def predictionAccuracy(weights1, weights2, theta_hidden, theta_softmax, testData, targetLabel):
    numCasesPred = shape(testData)[1]
    testData = concatenate((ones((1,numCasesPred)), testData), axis = 0)
    hidden_sum_L1 = dot(weights1, testData)
    relu_mask_hidden1 = ones(shape(hidden_sum_L1)) * (hidden_sum_L1>0)
    hidden_activation_L1 = hidden_sum_L1*relu_mask_hidden1
    hidden_activation_L1 = concatenate((ones((1,numCasesPred)), hidden_activation_L1), axis=0)
    hidden_sum_L2 = dot(weights2, hidden_activation_L1)
    relu_mask_hidden2 = ones(shape(hidden_sum_L2)) * (hidden_sum_L2>0)
    hidden_activation_L2 = hidden_sum_L2*relu_mask_hidden2
    hidden_activation_L2 = concatenate((ones((1,numCasesPred)), hidden_activation_L2), axis=0)
    hidden_sum_L3 = dot(theta_hidden, hidden_activation_L2)
    relu_mask_hidden3 = ones(shape(hidden_sum_L3)) * (hidden_sum_L3>0)
    hidden_activation_L3 = hidden_sum_L3*relu_mask_hidden3
    hidden_sum_softmax = dot(theta_softmax, hidden_activation_L3)
    hidden_sum_softmax = hidden_sum_softmax - hidden_sum_softmax.max(axis = 0)
    predictions = exp(hidden_sum_softmax)
    predictions = predictions / predictions.sum(axis = 0)
    pred = predictions.argmax(axis=0)
    accuracy = mean(pred == targetLabel) * 100
    print "Accuracy: ", accuracy, "%"
    scipy.io.savemat("predicting2and1layersNoFineTune.mat", mdict={'probability':predictions, 'prediction':pred,'labels':targetLabel})
    return accuracy


