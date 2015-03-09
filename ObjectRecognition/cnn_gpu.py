# TODO: Implement overlaping max-pooling, Implement Local Contrast Normalization, Implement square root sum squares pooling
# TODO: Look into Baysian Optimization for choosing parameters
# TODO: Implement dropout

from numpy import *
from scipy.signal import *
from linearDecoderReLU import *
from mlp_softmax_gpuReLU import *
import scipy.io
import time

def cnn_fineTune_costfunc(x, *args):
    trainImages, numClasses, num_kernels, l1Size, l2Size, l3Size, poolDim, patchDim, ZCAWhite, meanPatch, lambda_softmax, lambda_hidden, labels, groundTruth, dropout_probability, Batch, Set= args
    numImages = shape(trainImages)[3]
    imageDim = shape(trainImages)[0]
    imageChannels = shape(trainImages)[2]
    convolveDim = imageDim-patchDim+1
    pooledDim = int(floor(convolveDim/poolDim))
    inputSize = patchDim * patchDim * imageChannels
    convolvedFeatures = zeros((num_kernels, numImages, imageDim-patchDim+1, imageDim-patchDim+1))
    poolFeatures = zeros((num_kernels, numImages, floor(convolveDim/poolDim), floor(convolveDim/poolDim)))
    num_weights_kernel = num_kernels * (inputSize + 1)
    num_weights_L1 = l1Size * (pooledDim * pooledDim * num_kernels + 1)
    num_weights_L2 = l2Size * (l1Size + 1)
    num_weights_L3 = l3Size * (l2Size + 1)
    num_weights_softmax = numClasses * l3Size
    theta_kernels = reshape(x[0:num_weights_kernel], (num_kernels, inputSize + 1))
    theta_L1 = gpu.garray(reshape(x[num_weights_kernel:num_weights_kernel+num_weights_L1], (l1Size, pooledDim * pooledDim * num_kernels + 1)))
    theta_L2 = gpu.garray(reshape(x[num_weights_kernel+num_weights_L1:num_weights_kernel+num_weights_L1+num_weights_L2], (l2Size, l1Size + 1)))
    theta_L3 = gpu.garray(reshape(x[num_weights_kernel+num_weights_L1+num_weights_L2:num_weights_kernel+num_weights_L1+num_weights_L2+num_weights_L3], (l3Size, l2Size + 1)))
    theta_softmax = gpu.garray(reshape(x[num_weights_kernel+num_weights_L1+num_weights_L2+num_weights_L3:shape(x)[0]], (numClasses, l3Size)))
    theta_kernels_grad = gpu.zeros(shape(theta_kernels))
    theta_L1_grad = gpu.zeros(shape(theta_L1))
    theta_L2_grad = gpu.zeros(shape(theta_L2))
    theta_L3_grad = gpu.zeros(shape(theta_L3))
    theta_softmax_grad = gpu.zeros(shape(theta_softmax))
    dropout_l1 = gpu.garray(bernoulli.rvs(dropout_probability, size = (l1Size+1, numImages)))
    dropout_l2 = gpu.garray(bernoulli.rvs(dropout_probability, size = (l2Size+1, numImages)))
    dropout_l3 = gpu.garray(bernoulli.rvs(dropout_probability, size = (l3Size, numImages)))
    kernelWeights = theta_kernels[:,1:shape(theta_kernels)[1]]
    kernelBias = theta_kernels[:,0]
    whiteningKernelWeights = dot(kernelWeights, ZCAWhite)
    #whiteningKernelWeights = kernelWeights
    whiteningKernelBias = kernelBias - squeeze(dot(whiteningKernelWeights, meanPatch.transpose()))
    #whiteningKernelBias = kernelBias
    pred_cost = 0.0
    cum_accuracy = 0.0
    for imageNum in range(numImages):
        print "Now convolving image", imageNum+1, "in Batch", Batch+1, 
        for kernelNum in range(num_kernels):
            convolvedImage = zeros((imageDim-patchDim+1, imageDim-patchDim+1))
            for channel in range(3):
                rowKernelWeight = whiteningKernelWeights[kernelNum, channel * whiteningKernelWeights.shape[1]/imageChannels : (channel + 1) * whiteningKernelWeights.shape[1]/imageChannels]
                kernelWeight = reshape(rowKernelWeight, (sqrt(rowKernelWeight.shape[0]), sqrt(rowKernelWeight.shape[0])))
                kernelWeight = flipud(fliplr(squeeze(kernelWeight)))
                image = squeeze(trainImages[:,:,channel,imageNum])
                convolvedImage = convolvedImage + fftconvolve(image, kernelWeight, mode='valid')
            convolvedImage = convolvedImage + tile(whiteningKernelBias[kernelNum], shape(convolvedImage))
            relu_mask_convolved = ones(shape(convolvedImage)) * (convolvedImage>0)
            convolvedImage = convolvedImage*relu_mask_convolved
            convolvedFeatures[kernelNum, imageNum,:,:] = convolvedImage
            print "Now pooling kernel", kernelNum+1, "for image", imageNum+1, "in set", Set+1, "from Batch", Batch + 1
            for row in range(pooledDim):
                for col in range(pooledDim):
                   poolFeatures[kernelNum, imageNum, row, col] = sum(convolvedImage[row*poolDim:(row+1)*poolDim, col*poolDim:(col+1)*poolDim])/(poolDim*poolDim) 
        print "Now passing the pooled features into the fully-connected layers...."
        singlePooledFeatures = poolFeatures[:,imageNum,:,:]
        #singlePooledFeatures = transpose(singlePooledFeatures, (0,2,3,1))
        singlePooledFeatures_row = reshape(singlePooledFeatures, (product(singlePooledFeatures.shape), 1))
        inputs = gpu.garray(singlePooledFeatures_row)
        inputs = gpu.concatenate((gpu.ones((1, 1)), inputs), axis = 0)
        hidden_sum_L1 = gpu.dot(theta_L1, inputs)
        relu_mask_hidden1 = gpu.ones(shape(hidden_sum_L1)) * (hidden_sum_L1>0)
        hidden_activation_L1 = hidden_sum_L1*relu_mask_hidden1
        hidden_derivative_L1 = relu_mask_hidden1
        hidden_derivative_L1 = gpu.concatenate((gpu.ones((1, 1)), hidden_derivative_L1), axis=0)
        hidden_activation_L1 = gpu.concatenate((gpu.ones((1, 1)), hidden_activation_L1), axis=0) 
        hidden_sum_L2 = gpu.dot(theta_L2, hidden_activation_L1)
        relu_mask_hidden2 = gpu.ones(shape(hidden_sum_L2)) * (hidden_sum_L2>0)
        hidden_activation_L2 = hidden_sum_L2*relu_mask_hidden2
        hidden_derivative_L2 = relu_mask_hidden2
        hidden_derivative_L2 = gpu.concatenate((gpu.ones((1, 1)), hidden_derivative_L2), axis=0)
        hidden_activation_L2 = gpu.concatenate((gpu.ones((1, 1)), hidden_activation_L2), axis=0) 
        hidden_sum_L3 = gpu.dot(theta_L3, hidden_activation_L2)
        relu_mask_hidden3 = gpu.ones(shape(hidden_sum_L3)) * (hidden_sum_L3>0)
        hidden_derivative_L3 = relu_mask_hidden3
        hidden_activation_L3 = hidden_sum_L3*relu_mask_hidden3
        hidden_sum_softmax = gpu.dot(theta_softmax, hidden_activation_L3)
        hidden_sum_softmax = hidden_sum_softmax - hidden_sum_softmax.max(axis = 0)
        predictions = hidden_sum_softmax.exp()
        predictions = predictions / gpu.sum(predictions,axis = 0)
        pred = predictions.argmax(axis=0) + 1
        print "Performing predictions...."
        accuracy = mean(pred == labels[imageNum,:]) * 100
        cum_accuracy += accuracy
        print "Prediction:", pred
        print "Actual:", labels[imageNum,:]
        print "Total Accuracy is:", cum_accuracy/(imageNum+1), "% out of", imageNum+1, "images"
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
        current_cost = -1*sum(temp)
        pred_cost += -1*sum(temp)
        l2norm_cost = 0.5 * lambda_hidden*(gpu.sum(regularized_penalty_L3) + gpu.sum(regularized_penalty_L2) + gpu.sum(regularized_penalty_L1)) + 0.5 * lambda_softmax * gpu.sum(theta_softmax*theta_softmax)
        cost = pred_cost + l2norm_cost
        print "This image's prediction cost is", current_cost
        "Now performing backpropagation on CNN....."
        softmax_imd = reshape(groundTruth[:,imageNum].as_numpy_array(), shape(predictions)) - predictions.as_numpy_array()
        theta_softmax_grad += -1*gpu.dot(softmax_imd, gpu.garray(transpose(hidden_activation_L3.as_numpy_array()))) + lambda_softmax * theta_softmax
        deltaOut = -softmax_imd
        delta_L3_imd = gpu.dot(gpu.garray(transpose(theta_softmax.as_numpy_array())), deltaOut)
        delta_L3_imd2 = delta_L3_imd*hidden_derivative_L3
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
        delta_pool_imd = gpu.dot(gpu.garray(transpose(theta_L1.as_numpy_array())), delta_L1_imd2)
        delta_pool_imd2 = delta_pool_imd[1:shape(delta_pool_imd)[0]+1, :]
        delta_pool_2dMap = reshape(delta_pool_imd2.as_numpy_array(), (num_kernels, pooledDim, pooledDim))
        delta_pool_2dMap_upscaled = zeros((num_kernels, convolveDim, convolveDim))
        for kernelNum in range(num_kernels):
            delta_pool_2dMap_upscaled[kernelNum, :, :] = kron(delta_pool_2dMap[kernelNum,:,:], ones((poolDim, poolDim))) / (poolDim * poolDim)
            delta_pool_2dMap_upscaled[kernelNum, :, :] = delta_pool_2dMap_upscaled[kernelNum, :, :] * relu_mask_convolved
            delta_convolution = delta_pool_2dMap_upscaled[kernelNum, :, :]
            theta_kernels_grad_tmp = zeros((patchDim, patchDim))
            for channel in range(3):
                image = squeeze(trainImages[:,:,channel,imageNum])
                theta_kernels_grad_tmp = fftconvolve(image, delta_convolution, mode='valid')
                theta_kernels_grad[kernelNum, channel * patchDim * patchDim + 1: (channel + 1) * patchDim * patchDim + 1] += reshape(theta_kernels_grad_tmp, patchDim * patchDim)
            theta_kernels_grad[kernelNum, 0] += sum(delta_convolution)
    print "Total prediction cost:", pred_cost/numImages
    theta_kernels_grad[:,1:shape(theta_kernels_grad)[1]] = gpu.dot(theta_kernels_grad[:,1:shape(theta_kernels_grad)[1]], ZCAWhite)
    theta_kernels_grad[:,0] = theta_kernels_grad[:,0] - squeeze(gpu.dot(theta_kernels_grad[:,1:shape(theta_kernels_grad)[1]], meanPatch.transpose()).as_numpy_array())
    theta_L1_grad = theta_L1_grad/numImages
    theta_L2_grad = theta_L2_grad/numImages
    theta_L3_grad = theta_L3_grad/numImages
    theta_kernels_grad = theta_kernels_grad/numImages
    theta_L1_grad[:, 1:shape(theta_L1_grad)[1]] = theta_L1_grad[:, 1:shape(theta_L1_grad)[1]] + theta_L1[:, 1: shape(theta_L1)[1]] * lambda_hidden
    theta_L2_grad[:, 1:shape(theta_L2_grad)[1]] = theta_L2_grad[:, 1:shape(theta_L2_grad)[1]] + theta_L2[:, 1: shape(theta_L2)[1]] * lambda_hidden
    theta_L3_grad[:, 1:shape(theta_L3_grad)[1]] = theta_L3_grad[:, 1:shape(theta_L3_grad)[1]] + theta_L3[:, 1: shape(theta_L3)[1]] * lambda_hidden 
    theta_L1_grad = reshape(theta_L1_grad.as_numpy_array(), num_weights_L1)
    theta_L2_grad = reshape(theta_L2_grad.as_numpy_array(), num_weights_L2)
    theta_L3_grad = reshape(theta_L3_grad.as_numpy_array(), num_weights_L3)
    theta_kernels_grad = reshape(theta_kernels_grad.as_numpy_array(), num_weights_kernel)
    theta_softmax_grad = reshape(theta_softmax_grad.as_numpy_array(), num_weights_softmax)
    del inputs
    del theta_kernels
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
    del regularized_penalty_L1
    del regularized_penalty_L2
    gpu.free_reuse_cache()
    return pred_cost, hstack((theta_kernels_grad, theta_L1_grad,theta_L2_grad,theta_L3_grad,theta_softmax_grad))

def cnnConvolve(patchDim, numFeatures, images, W, b, ZCAWhite, meanPatch, numBatch, numSets):
    numImages = shape(images)[3]
    imageDim = shape(images)[0]
    imageChannels = shape(images)[2]
    convolvedFeatures = zeros((numFeatures, numImages, imageDim-patchDim+1, imageDim-patchDim+1))
    whiteningFeatures = dot(W, ZCAWhite)
    whiteningB = b - squeeze(dot(whiteningFeatures, meanPatch.transpose()))
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
                convolvedImage = convolvedImage + fftconvolve(im, feature, mode='valid')
            convolvedImage = convolvedImage + tile(whiteningB[featureNum], shape(convolvedImage))
            #print shape(convolvedImage)
            #convolvedImage = 1/(1 + exp(-convolvedImage))
            relu_mask_convolved = ones(shape(convolvedImage)) * (convolvedImage>0)
            convolvedImage = convolvedImage*relu_mask_convolved
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

def ZCAWhiteningGPU(patches, epsilon):
    print "Applying ZCA Whitening..."
    numPatches = shape(patches)[1]
    meanPatch = mean(patches, axis = 1)
    patches = transpose(patches)-meanPatch
    patches = transpose(patches)
    patches = gpu.garray(patches)
    sigma = gpu.dot(patches, gpu.garray(transpose(patches.as_numpy_array())))/numPatches
    U,S,V = linalg.svd(sigma.as_numpy_array())
    del sigma
    gpu.free_reuse_cache()
    U = gpu.garray(U)
    S = gpu.garray(S)
    ZCAWhiteTemp = gpu.dot(U, (1/(S+epsilon).sqrt()).diag())
    ZCAWhite = gpu.dot(ZCAWhiteTemp, gpu.garray(transpose(U.as_numpy_array())))
    patches = gpu.dot(ZCAWhite, patches)
    scipy.io.savemat('patchesZCA', mdict={'patchesZCA':patches.as_numpy_array()})
    print "Finished Whitening."
    scipy.io.savemat('whitening.mat', mdict={'ZCAWhite': ZCAWhite.as_numpy_array(), 'meanPatch':meanPatch})
    del U
    del S
    del ZCAWhiteTemp
    del ZCAWhite
    del patches
    gpu.free_reuse_cache()
    patches = scipy.io.loadmat('patchesZCA')['patchesZCA']
    ZCAWhite = scipy.io.loadmat('whitening.mat')['ZCAWhite']
    return patches, ZCAWhite, meanPatch

def imageSampling(images, patchDim, imageChannels):
    num_patches = 100000
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

def cnn_learn(patchDim, poolDim, num_hidden, unLabeledImages, imageChannels, epsilon):
    patches = imageSampling(unLabeledImages, patchDim, imageChannels)
    #patches, ZCAWhite, meanPatch = ZCAWhiteningGPU(patches, epsilon)
    num_input = patchDim * patchDim * imageChannels
    num_features = num_hidden
    gpu.free_reuse_cache()
    learntFeatures = linear_decoder_run_gpu(patches, num_input, num_features)
    return learntFeatures

def cnn_learn_ReLU(patchDim, poolDim, num_hidden, unLabeledImages, imageChannels, epsilon):
    patches = imageSampling(unLabeledImages, patchDim, imageChannels)
    patches, ZCAWhite, meanPatch = ZCAWhitening(patches, epsilon)
    num_input = patchDim * patchDim * imageChannels
    num_features = num_hidden
    print num_input
    print num_features
    learntFeatures = linear_decoder_run_ReLU(patches, num_input, num_features)
    return learntFeatures

def cnn_getFeatures(patchDim, poolDim, numKernels, imageChannels):
    num_hidden = numKernels
    epsilon = 0.1
    unLabeledData = scipy.io.loadmat('unlabeledData.mat')
    unLabeledImages = unLabeledData['data']
    result = cnn_learn_ReLU(patchDim, poolDim, num_hidden, unLabeledImages,imageChannels, epsilon)
    return result

def cnn_testFineTune(learntFeatures):
    patchDim = 9
    poolDim = 8
    num_hidden = 100
    data = scipy.io.loadmat('trainBatches.mat')
    trainImages = data['X']
    trainLabels = data['y']
    imgDim = shape(trainImages[0,:,:,:,:])[0]
    convolveDim = imgDim-patchDim+1
    pooledDim = int(floor(convolveDim/poolDim))
    inputSize = patchDim * patchDim * 3
    pooledInputSize = pooledDim * pooledDim * num_hidden 
    l1Size = floor(pooledInputSize/6)
    l2Size = floor(l1Size/5)
    l3Size = floor(l2Size/2)
    lambda_softmax = 1e-4
    lambda_hidden = 2e-5
    numClasses = trainLabels[0,:,:].max()
    num_weights_kernel = num_hidden * (inputSize + 1)
    num_weights_L1 = l1Size * (pooledInputSize + 1)
    num_weights_L2 = l2Size * (l1Size + 1)
    num_weights_L3 = l3Size * (l2Size + 1)
    num_weights_softmax = numClasses * l3Size
    #r = sqrt(6)/sqrt(num_hidden + pooledDim * pooledDim * num_hidden+l1Size+l2Size+l3Size+1)
    theta_kernels = learntFeatures
    #theta_kernels = (random.rand(num_hidden, inputSize + 1))*2*r-r
    #theta_L1 = (random.rand(l1Size, pooledDim * pooledDim * num_hidden + 1))*2*r-r
    #theta_L2 = (random.rand(l2Size, l1Size+1))*2*r-r
    #theta_L3 = (random.rand(l3Size, l2Size+1))*2*r-r
    #theta_softmax = (random.rand(numClasses, l3Size))*2*r-r
    theta_kernels = reshape(theta_kernels, num_weights_kernel)
    #theta_L1 = reshape(theta_L1, num_weights_L1)
    #theta_L2 = reshape(theta_L2, num_weights_L2)
    #theta_L3 = reshape(theta_L3, num_weights_L3)
    #theta_softmax = reshape(theta_softmax, num_weights_softmax)
    theta_mlpsoftmax = squeeze(scipy.io.loadmat('cnnMlpSoftmaxTrain.mat')['theta'])
    #theta = hstack((theta_kernels, theta_L1,theta_L2,theta_L3,theta_softmax))
    theta = hstack((theta_kernels, theta_mlpsoftmax))
    image_steSize = 200
    for i in range(shape(trainImages)[0]):
        print "Training Batch", i
        trainImg = trainImages[i,:,:,:,:]
        numImg = shape(trainImg)[3]
        imgDim = shape(trainImg)[0]
        numFeatures = num_hidden
        ZCA = scipy.io.loadmat('whitening.mat')
        ZCAWhite = ZCA['ZCAWhite']
        meanPatch = ZCA['meanPatch']
        for j in range(numImg/image_steSize):
            trainImages_thisBatch = trainImg[:,:,:,j*image_steSize:(j+1)*image_steSize]
            labels = trainLabels[i,j*image_steSize:(j+1)*image_steSize,:]
            groundTruth = ground_Truth(labels,image_steSize)
            args = (trainImages_thisBatch, numClasses, numFeatures, l1Size, l2Size, l3Size, poolDim, patchDim, ZCAWhite, meanPatch, lambda_softmax, lambda_hidden, labels, groundTruth, 1.0, i , j)
            print "Starting CNN Training..."
            opttheta = optimize.fmin_l_bfgs_b(cnn_fineTune_costfunc, theta, args=args, maxiter=200)
            theta = opttheta[0]
            print "Training Batch",i, "finished."
        scipy.io.savemat('cnnMlpSoftmaxTrainFineTune.mat', mdict={'theta': theta})
    return theta
    
def cnn_train(learntFeatures):
    patchDim = 9
    poolDim = 8
    num_hidden = 100
    data = scipy.io.loadmat('trainBatches.mat')
    trainImages = data['X']
    trainLabels = data['y']
    imgDim = shape(trainImages[0,:,:,:,:])[0]
    inputSize = floor((imgDim-patchDim+1)/poolDim) * floor((imgDim-patchDim+1)/poolDim) * num_hidden
    l1Size = floor(inputSize/6)
    l2Size = floor(l1Size/5)
    l3Size = floor(l2Size/2)
    lambda_softmax = 1e-4
    lambda_hidden = 2e-5
    numClasses = trainLabels[0,:,:].max()
    num_weights_L1 = l1Size * (inputSize + 1)
    num_weights_L2 = l2Size * (l1Size + 1)
    num_weights_L3 = l3Size * (l2Size + 1)
    num_weights_softmax = numClasses * l3Size
    r = sqrt(6)/sqrt(inputSize+l1Size+l2Size+l3Size+1)
    theta_kernels = learntFeatures
    theta_L1 = (random.rand(l1Size, inputSize+1))*2*r-r
    theta_L2 = (random.rand(l2Size, l1Size+1))*2*r-r
    theta_L3 = (random.rand(l3Size, l2Size+1))*2*r-r
    theta_softmax = (random.rand(numClasses, l3Size))*2*r-r
    theta_L1 = reshape(theta_L1, num_weights_L1)
    theta_L2 = reshape(theta_L2, num_weights_L2)
    theta_L3 = reshape(theta_L3, num_weights_L3)
    theta_softmax = reshape(theta_softmax, num_weights_softmax)
    theta = hstack((theta_L1, theta_L2, theta_L3, theta_softmax))
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
        args = (numClasses, inputSize, l1Size, l2Size, l3Size, lambda_softmax, lambda_hidden, inputs, labels, groundTruth, 1.0)
        print "Starting Softmax Training..."
        opttheta = optimize.fmin_l_bfgs_b(mlpSoftmax_costfunc, theta, args=args, maxiter=400)
        theta = opttheta[0]
        print "Training Batch",i, "finished."
        scipy.io.savemat('cnnMlpSoftmaxTrain.mat', mdict={'theta': theta})
    return theta
    
def cnn_predict():
    patchDim = 9
    poolDim = 8
    num_hidden = 100
    data = scipy.io.loadmat('testBatch.mat')
    testImages = data['X']
    testLabels = data['y']
    numImg = shape(testImages)[3]
    imgDim = shape(testImages)[0]
    inputSize = floor((imgDim-patchDim+1)/poolDim) * floor((imgDim-patchDim+1)/poolDim) * num_hidden
    l1Size = floor(inputSize/6)
    l2Size = floor(l1Size/5)
    l3Size = floor(l2Size/2)
    numClasses = testLabels.max()
    num_weights_L1 = l1Size * (inputSize + 1)
    num_weights_L2 = l2Size * (l1Size + 1)
    num_weights_L3 = l3Size * (l2Size + 1)
    #num_weights_softmax = numClasses * l2Size
    numFeatures = num_hidden
    ZCA = scipy.io.loadmat('whitening.mat')
    ZCAWhite = ZCA['ZCAWhite']
    meanPatch = ZCA['meanPatch']
    #learntFeatures = scipy.io.loadmat('learntFeaturesReLU.mat')['learntFeatures']
    #mlpTheta = squeeze(scipy.io.loadmat('cnnMlpSoftmaxTrainBatch3.mat')['theta'])
    allFeatures = squeeze(scipy.io.loadmat('cnnMlpSoftmaxTrainFineTuneBatch2.mat')['theta'])
    learntFeatures = reshape(allFeatures[0:num_hidden*(patchDim*patchDim*3+1)], (num_hidden, patchDim*patchDim*3+1))
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
    theta = allFeatures[num_hidden*(patchDim*patchDim*3+1):shape(allFeatures)[0]]
    #theta = mlpTheta
    print "Now testing prediction accuracy..."
    theta_L1 = reshape(theta[0:num_weights_L1], (l1Size, inputSize + 1))
    theta_L2 = reshape(theta[num_weights_L1:num_weights_L2+num_weights_L1], (l2Size, l1Size + 1))
    theta_L3 = reshape(theta[num_weights_L2+num_weights_L1:num_weights_L3+num_weights_L2+num_weights_L1], (l3Size, l2Size + 1))
    theta_softmax = reshape(theta[num_weights_L3+num_weights_L2+num_weights_L1:shape(theta)[0]], (numClasses, l3Size))
    numCasesPred = shape(inputs)[1]
    inputs = concatenate((ones((1,numCasesPred)), inputs), axis = 0)
    hidden_sum_L1 = dot(theta_L1, inputs)
    relu_mask_hidden1 = ones(shape(hidden_sum_L1)) * (hidden_sum_L1>0)
    hidden_activation_L1 = hidden_sum_L1*relu_mask_hidden1
    hidden_activation_L1 = concatenate((ones((1,numCasesPred)), hidden_activation_L1), axis=0)
    hidden_sum_L2 = dot(theta_L2, hidden_activation_L1)
    relu_mask_hidden2 = ones(shape(hidden_sum_L2)) * (hidden_sum_L2>0)
    hidden_activation_L2 = hidden_sum_L2*relu_mask_hidden2
    hidden_activation_L2 = concatenate((ones((1,numCasesPred)), hidden_activation_L2), axis=0) 
    hidden_sum_L3 = dot(theta_L3, hidden_activation_L2)
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
    
def run_test():
    patchDim = 16
    poolDim = 9
    num_hidden = 150
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
    print inputSize
    print l1Size
    print l2Size
    print numImg
    opttheta = optimize.fmin_l_bfgs_b(mlpSoftmax_costfunc, theta, args=args, maxiter=300)
    theta = opttheta[0]
    return theta
    
def cnnSoftmax_test():
    data = scipy.io.loadmat('learntFeaturesGPU.mat')
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
#cnn_getFeatures()
#learntFeatures = scipy.io.loadmat('learntFeaturesReLU.mat')['learntFeatures']
#cnn_train(learntFeatures)
