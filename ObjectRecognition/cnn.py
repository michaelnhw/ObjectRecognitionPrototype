# TODO: Implement overlaping max-pooling, Implement Local Contrast Normalization, Implement square root sum squares pooling
# TODO: Look into Baysian Optimization for choosing parameters
# TODO: Implement dropout

from numpy import *
from scipy.signal import *
from linearDecoder import *
from mlp_softmax import *
import scipy.io
import time

def cnnConvolve(patchDim, numFeatures, images, W, b, ZCAWhite, meanPatch, numBatch, numSets):
    numImages = shape(images)[3]
    imageDim = shape(images)[0]
    imageChannels = shape(images)[2]
    convolvedFeatures = zeros((numFeatures, numImages, imageDim-patchDim+1, imageDim-patchDim+1))
    whiteningFeatures = dot(W, ZCAWhite)
    whiteningB = b - squeeze(dot(whiteningFeatures, meanPatch))
    for imageNum in range(numImages):
        print "Now convolving image", imageNum, "in Set", numSets, ", Batch", numBatch
        for featureNum in range(numFeatures):
            #print "Convolving feature", featureNum
            convolvedImage = zeros((imageDim-patchDim+1, imageDim-patchDim+1))
            for channel in range(3):
                rowFeatures = whiteningFeatures[featureNum, channel*W.shape[1]/imageChannels: (channel+1)*W.shape[1]/imageChannels]
                feature = reshape(rowFeatures, (sqrt(rowFeatures.shape[0]), sqrt(rowFeatures.shape[0])))
                feature = flipud(fliplr(squeeze(feature)))
                im = squeeze(images[:,:,channel, imageNum])
                convolvedImage = convolvedImage + convolve2d(im, feature, 'valid')
            convolvedImage = convolvedImage + tile(whiteningB[featureNum], shape(convolvedImage))
            convolvedImage = 1/(1 + exp(-convolvedImage))
            convolvedFeatures[featureNum, imageNum,:,:] = convolvedImage
    print "Convolution completed."
    return convolvedFeatures

def cnnPool(poolDim, convolvedFeatures, numBatch, imgSets):
    numImages = shape(convolvedFeatures)[1]
    numFeatures = shape(convolvedFeatures)[0]
    convolveDim = shape(convolvedFeatures)[2]
    poolFeatures = zeros((numFeatures, numImages, floor(convolveDim/poolDim), floor(convolveDim/poolDim)))
    numSets = int(floor(convolveDim/poolDim))
    for imageNum in range(numImages):
        print "Now performing mean-pooling on image", imageNum, "in Set", imgSets, ", Batch", numBatch
        for featureNum in range(numFeatures):
            for row in range(numSets):
                for col in range(numSets):
                    poolFeatures[featureNum, imageNum, row, col] = sum(convolvedFeatures[featureNum, imageNum, row*poolDim:(row+1)*poolDim, col*poolDim:(col+1)*poolDim])/(poolDim*poolDim)
    print "Pooling Completed."
    return poolFeatures
    
def cnnMaxPool_nonoverlaping(poolDim, convolvedFeatures, numBatch, imgSets):
    numImages = shape(convolvedFeatures)[1]
    numFeatures = shape(convolvedFeatures)[0]
    convolveDim = shape(convolvedFeatures)[2]
    poolFeatures = zeros((numFeatures, numImages, floor(convolveDim/poolDim), floor(convolveDim/poolDim)))
    numSets = int(floor(convolveDim/poolDim))
    for imageNum in range(numImages):
        print "Now performing max-pooling on image", imageNum, "in Set", imgSets, ", Batch", numBatch
        for featureNum in range(numFeatures):
            for row in range(numSets):
                for col in range(numSets):
                    poolFeatures[featureNum, imageNum, row, col] = (convolvedFeatures[featureNum, imageNum, row*poolDim:(row+1)*poolDim, col*poolDim:(col+1)*poolDim]).max()
    print "Max-Pooling Completed."
    return poolFeatures
    
def cnnMaxPool_overlaping(poolDim, convolvedFeatures, numBatch, imgSets):
    numImages = shape(convolvedFeatures)[1]
    numFeatures = shape(convolvedFeatures)[0]
    convolveDim = shape(convolvedFeatures)[2]
    poolFeatures = zeros((numFeatures, numImages, (convolveDim-poolDim+1), (convolveDim-poolDim+1)))
    numSets = (convolveDim-poolDim+1)
    for imageNum in range(numImages):
        print "Now performing max-pooling-with-overlapping on image", imageNum, "in Set", imgSets, ", Batch", numBatch
        for featureNum in range(numFeatures):
            for row in range(numSets):
                for col in range(numSets):
                    poolFeatures[featureNum, imageNum, row, col] = max(convolvedFeatures[featureNum, imageNum, row*poolDim:(row+1)*poolDim, col*poolDim:(col+1)*poolDim])
    print "Max-Pooling Completed."
    return poolFeatures

def ZCAWhitening(patches, epsilon):
    print "Applying ZCA Whitening..."
    numPatches = shape(patches)[1]
    meanPatch = mean(patches, axis = 1)
    patches = transpose(patches)-meanPatch
    patches = transpose(patches)
    sigma = dot(patches, transpose(patches))/numPatches
    U,S,V = linalg.svd(sigma)
    ZCAWhiteTemp = dot(U, diag(1/sqrt(S+epsilon)))
    ZCAWhite = dot(ZCAWhiteTemp, transpose(U))
    patches = dot(ZCAWhite, patches)
    scipy.io.savemat('patchesZCA', mdict={'patchesZCA':patches})
    print "Finished Whitening."
    scipy.io.savemat('whitening.mat', mdict={'ZCAWhite': ZCAWhite, 'meanPatch':meanPatch})
    return patches, ZCAWhite, meanPatch

def imageSampling(images, patchDim, imageChannels):
    num_patches = 300000
    print "Sampling image patches with size", num_patches
    patches = zeros((patchDim*patchDim*imageChannels, num_patches))
    for i in range(num_patches):
        print "Obtaining patch", i
        pick_img = round(random.uniform(0,1) * shape(images)[3])
        if pick_img == shape(images)[3]:
            pick_img = shape(images)[3]-1
        pick_row = round(random.uniform(0,1) * (shape(images)[0]- patchDim))
        pick_col = round(random.uniform(0,1) * (shape(images)[1]- patchDim))
        for channel in range(imageChannels):
            patch = images[pick_row:(pick_row+patchDim), pick_col:(pick_col+patchDim),channel, pick_img]
            patches[channel*shape(patches)[0]/imageChannels:(channel+1)*shape(patches)[0]/imageChannels,i] = hstack((patch.T))
    print "Obtained patches."
    patches = patches/255
    scipy.io.savemat('patches', mdict={'patches':patches})
    return patches

def cnn_learn(patchDim, poolDim, num_hidden, unLabeledImages, imageChannels, epsilon):
    patches = imageSampling(unLabeledImages, patchDim, imageChannels)
    patches, ZCAWhite, meanPatch = ZCAWhitening(patches, epsilon)
    num_input = patchDim * patchDim * imageChannels
    num_features = num_hidden
    learntFeatures = linear_decoder_run(patches, num_input, num_features)
    return learntFeatures

def cnn_learn_ReLU(patchDim, poolDim, num_hidden, unLabeledImages, imageChannels, epsilon):
    patches = imageSampling(unLabeledImages, patchDim, imageChannels)
    patches, ZCAWhite, meanPatch = ZCAWhitening(patches, epsilon)
    num_input = patchDim * patchDim * imageChannels
    num_features = num_hidden
    learntFeatures = linear_decoder_run_ReLU(patches, num_input, num_features)
    return learntFeatures

def cnn_getFeatures():
    patchDim = 8
    poolDim = 29
    num_hidden = 400
    imageChannels = 3
    epsilon = 0.1
    unLabeledData = scipy.io.loadmat('unlabeledData.mat')
    unLabeledImages = unLabeledData['data']
    result = cnn_learn(patchDim, poolDim, num_hidden, unLabeledImages,imageChannels, epsilon)
    return result
    
def cnn_train(learntFeatures):
    patchDim = 8
    poolDim = 29
    num_hidden = 400
    subBatchSize = 100
    data = scipy.io.loadmat('trainBatches.mat')
    trainImages = data['X']
    trainLabels = data['y']
    imgDim = shape(trainImages[0,:,:,:,:])[0]
    inputSize = floor((imgDim-patchDim+1)/poolDim) * floor((imgDim-patchDim+1)/poolDim) * num_hidden
    l1Size = floor(inputSize/6)
    l2Size = floor(l1Size/5)
    lambda_softmax = 1e-4
    lambda_hidden = 2e-5
    numClasses = trainLabels[0,:,:].max()
    num_weights_L1 = l1Size * (inputSize + 1)
    num_weights_L2 = l2Size * (l1Size + 1)
    num_weights_softmax = numClasses * l2Size
    r = sqrt(6)/sqrt(inputSize+l1Size+l2Size+1)
    theta_L1 = (random.rand(l1Size, inputSize+1))*2*r-r
    theta_L2 = (random.rand(l2Size, l1Size+1))*2*r-r
    theta_softmax = (random.rand(numClasses, l2Size))*2*r-r
    theta_L1 = reshape(theta_L1, num_weights_L1)
    theta_L2 = reshape(theta_L2, num_weights_L2)
    theta_softmax = reshape(theta_softmax, num_weights_softmax)
    theta = hstack((theta_L1, theta_L2, theta_softmax))
    for i in range(shape(trainImages)[0]):
        print "Training Batch", i
        trainImg = trainImages[i,:,:,:,:]
        numImg = shape(trainImg)[3]
        imgDim = shape(trainImg)[0]
        numFeatures = num_hidden
        ZCA = scipy.io.loadmat('whitening.mat')
        ZCAWhite = ZCA['ZCAWhite']
        meanPatch = ZCA['meanPatch']
        W = learntFeatures[:,1:shape(learntFeatures)[1]]
        b = learntFeatures[:,0]
        stepSize = 50
        pooledFeatures = zeros((numFeatures, numImg, floor((imgDim-patchDim+1)/poolDim), floor((imgDim-patchDim+1)/poolDim)))
        for j in range(numFeatures/stepSize):
            feature_start = j * stepSize
            feature_end = (j+1) * stepSize
            Wt = W[feature_start:feature_end,:]
            bt = b[feature_start:feature_end]
            convolveThis = cnnConvolve(patchDim, stepSize, trainImg, Wt, bt, ZCAWhite, meanPatch, i, j)
            poolThis = cnnPool(poolDim, convolveThis, i, j)
            pooledFeatures[feature_start:feature_end,:,:,:] = poolThis
            del convolveThis
            del poolThis
        scipy.io.savemat('pooledFeaturesTrain.mat', mdict={'pooledFeaturesTrain': pooledFeatures})
        pooledFeatures = transpose(pooledFeatures, (0,2,3,1))
        inputs = reshape(pooledFeatures, (product(pooledFeatures.shape)/numImg, numImg))
        labels = trainLabels[i,:,:]
        numCases = shape(inputs)[1]
        groundTruth = ground_Truth(labels,numCases)
        args = (numClasses, inputSize, l1Size, l2Size, lambda_softmax, lambda_hidden, inputs, labels, groundTruth)
        print "Starting Softmax Training..."
        opttheta = optimize.fmin_l_bfgs_b(mlpSoftmax_costfunc, theta, fprime=mlpSoftmax_grad, args=args, maxiter=450)
        theta = opttheta[0]
        print "Training Batch",i, "finished."
        scipy.io.savemat('cnnMlpSoftmaxTrain.mat', mdict={'theta': theta})
    return theta
    
def cnn_predict(learntFeatures, opttheta):
    patchDim = 8
    poolDim = 29
    num_hidden = 400
    data = scipy.io.loadmat('testBatch.mat')
    testImages = data['X']
    testLabels = data['y']
    numImg = shape(testImages)[3]
    imgDim = shape(testImages)[0]
    inputSize = floor((imgDim-patchDim+1)/poolDim) * floor((imgDim-patchDim+1)/poolDim) * num_hidden
    l1Size = floor(inputSize/6)
    l2Size = floor(l1Size/5)
    numClasses = testLabels.max()
    num_weights_L1 = l1Size * (inputSize + 1)
    num_weights_L2 = l2Size * (l1Size + 1)
    #num_weights_softmax = numClasses * l2Size
    numFeatures = num_hidden
    ZCA = scipy.io.loadmat('whitening.mat')
    ZCAWhite = ZCA['ZCAWhite']
    meanPatch = ZCA['meanPatch']
    W = learntFeatures[:,1:shape(learntFeatures)[1]]
    b = learntFeatures[:,0]
    stepSize = 10
    pooledFeatures = zeros((numFeatures, numImg, floor((imgDim-patchDim+1)/poolDim), floor((imgDim-patchDim+1)/poolDim)))
    i = "Test Batch"
    for j in range(numFeatures/stepSize):
        feature_start = j * stepSize
        feature_end = (j+1) * stepSize
        Wt = W[feature_start:feature_end,:]
        bt = b[feature_start:feature_end]
        convolveThis = cnnConvolve(patchDim, stepSize, testImages, Wt, bt, ZCAWhite, meanPatch, i, j)
        poolThis = cnnPool(poolDim, convolveThis, i, j)
        pooledFeatures[feature_start:feature_end,:,:,:] = poolThis
        del convolveThis
        del poolThis
    scipy.io.savemat('pooledFeaturesTest.mat', mdict={'pooledFeaturesTest': pooledFeatures})
    pooledFeatures = transpose(pooledFeatures, (0,2,3,1))
    inputs = reshape(pooledFeatures, (product(pooledFeatures.shape)/numImg, numImg))
    theta = opttheta
    print "Now testing prediction accuracy..."
    theta_L1 = reshape(theta[0:num_weights_L1], (l1Size, inputSize + 1))
    theta_L2 = reshape(theta[num_weights_L1:num_weights_L2+num_weights_L1], (l2Size, l1Size + 1))
    theta_softmax = reshape(theta[num_weights_L2+num_weights_L1:shape(theta)[0]], (numClasses, l2Size))
    numCasesPred = shape(inputs)[1]
    inputs = concatenate((ones((1,numCasesPred)), inputs), axis = 0)
    hidden_sum_L1 = dot(theta_L1, inputs)
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
    
def run_test():
    patchDim = 8
    poolDim = 29
    num_hidden = 400
    data = scipy.io.loadmat('trainBatches.mat')
    trainImages = data['X']
    trainLabels = data['y']
    imgDim = shape(trainImages[0,:,:,:,:])[0]
    inputSize = floor((imgDim-patchDim+1)/poolDim) * floor((imgDim-patchDim+1)/poolDim) * num_hidden
    l1Size = floor(inputSize/6)
    l2Size = floor(l1Size/5)
    lambda_softmax = 7e-5
    lambda_hidden = 5e-5
    numClasses = trainLabels[0,:,:].max()
    num_weights_L1 = l1Size * (inputSize + 1)
    num_weights_L2 = l2Size * (l1Size + 1)
    num_weights_softmax = numClasses * l2Size
    r = sqrt(6)/sqrt(inputSize+l1Size+l2Size+1)
    theta_L1 = (random.rand(l1Size, inputSize+1))*2*r-r
    theta_L2 = (random.rand(l2Size, l1Size+1))*2*r-r
    theta_softmax = (random.rand(numClasses, l2Size))*2*r-r
    theta_L1 = reshape(theta_L1, num_weights_L1)
    theta_L2 = reshape(theta_L2, num_weights_L2)
    theta_softmax = reshape(theta_softmax, num_weights_softmax)
    theta = hstack((theta_L1, theta_L2, theta_softmax))
    trainImg = trainImages[0,:,:,:,:]
    numImg = shape(trainImg)[3]
    data = scipy.io.loadmat('pooledFeaturesTrain.mat')
    pooledFeatures = data['pooledFeaturesTrain']
    pooledFeatures = transpose(pooledFeatures, (0,2,3,1))
    inputs = reshape(pooledFeatures, (product(pooledFeatures.shape)/numImg, numImg))
    labels = trainLabels[0,:,:]
    numCases = shape(inputs)[1]
    groundTruth = ground_Truth(labels,numCases)
    args = (numClasses, inputSize, l1Size, l2Size, lambda_softmax, lambda_hidden, inputs, labels, groundTruth)
    print "Starting Softmax Training..."
    opttheta = optimize.fmin_l_bfgs_b(mlpSoftmax_costfunc, theta, fprime=mlpSoftmax_grad, args=args, maxiter=300)
    theta = opttheta[0]
    
def cnnSoftmax_test():
    data = scipy.io.loadmat('learntFeatures.mat')
    learntFeatures = data['learntFeatures']
    theta = cnn_train(learntFeatures)
    pred, testLabels = cnn_predict(learntFeatures, theta)
    return pred
    
def cnnSoftmax_pred():
    data = scipy.io.loadmat('learntFeatures.mat')
    learntFeatures = data['learntFeatures']
    theta = scipy.io.loadmat('cnnMlpSoftmaxTrain.mat')
    theta = theta['theta']
    pred, testLabels = cnn_predict(learntFeatures, theta)
    return pred
    

#start=time.clock()
#trainFeature = cnn_test()
#end = time.clock()
#print "Total running time:", end-sttart











