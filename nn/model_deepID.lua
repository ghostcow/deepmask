----------------------------------------------------------------------
-- DeepID - nn model for each patch
--
-- Input: (batchsize)XdXkX31 (d = 1/3,k = 31/39)
-- Output: 160d feature, classification into one of nLabels
----------------------------------------------------------------------
require 'nn'
require 'cunn'
require 'ccn2'
require 'torch'
require 'image'
require 'options'

-- NOTE : model for rgb patch of size 31x31
imageDim = 31
local inputDim = 3 -- number of input maps of 1st layer
featureDim = 160

-- filter sizes & number of maps for layers C1,C2,C3,C4
filtersSize = {4, 3, 3, 2}
numMaps = {32, 48, 64, 80} -- NOTE: original paper uses {20, 40, 60, 80}
maxPoolingSize = 2
maxPoolingStride = 2

-- layers ids (except the multi-scale layer)
layersIds = {C1=2,C2=5,C3=8, F6=12}
multiScaleLayerId = 11
-- F5 layer is divied into 2 parts in the multi scale layer
multiScaleTrainableLayerIds = {{3}, {1,5}}
----------------------------------------------------------------------
-- use -visualize to show network
-- parse command line arguments
if not opt then
    print '==> processing options'
    opt = getOptions()
end

----------------------------------------------------------------------
print '==> building model'

model = nn.Sequential()

-- convert from Torch batchSizeX3XheightXwidth format to ccn2 format 3XheightXwidthXbatchSize
model:add(nn.Transpose({1,4},{1,3},{1,2}))

local inputMapDim = imageDim -- dimension of input map to any layer
local outputMapDim 	     -- dimension of output map of any layer
local layerIndex = 1

-- C1,M1 & C2,M2 (convolution layers, each followed by max-pooling layer)
for iLayer = 1,2 do
    -- convolution layer
    model:add(ccn2.SpatialConvolution(inputDim, numMaps[layerIndex], filtersSize[layerIndex]))
    model:add(nn.ReLU())
    outputMapDim = inputMapDim - filtersSize[layerIndex] + 1
    print(string.format('C%d : %dx%dx%dx%d@%dx%d', iLayer,
        numMaps[layerIndex], filtersSize[layerIndex], filtersSize[layerIndex], inputDim, outputMapDim, outputMapDim))

    -- max-pooling layer
    inputMapDim = outputMapDim
    model:add(ccn2.SpatialMaxPooling(maxPoolingSize, maxPoolingStride))
    outputMapDim = inputMapDim / maxPoolingStride
    print(string.format('M%d : %dx%dx%dx%d@%dx%d', iLayer,
        numMaps[layerIndex], maxPoolingSize, maxPoolingSize, numMaps[layerIndex], outputMapDim, outputMapDim))

    layerIndex = layerIndex + 1
    inputMapDim = outputMapDim
    inputDim = numMaps[layerIndex - 1]
end

-- C3,M3 (locally connected layer, followed by max-pooling layer)
-- NOTE: original paper uses weights sharing over 2x2 regeions
inputDim = numMaps[layerIndex - 1]
model:add(ccn2.SpatialConvolutionLocal(inputDim, numMaps[layerIndex], inputMapDim, filtersSize[layerIndex]))
model:add(nn.ReLU())
outputMapDim = inputMapDim - filtersSize[layerIndex] + 1
print(string.format('C3 : %dx%dx%dx%d@%dx%d',
    numMaps[layerIndex], filtersSize[layerIndex], filtersSize[layerIndex], inputDim, outputMapDim, outputMapDim))

inputMapDim = outputMapDim
model:add(ccn2.SpatialMaxPooling(maxPoolingSize, maxPoolingStride))
outputMapDim = inputMapDim / maxPoolingStride
print(string.format('M3 : %dx%dx%dx%d@%dx%d',
    numMaps[layerIndex], maxPoolingSize, maxPoolingSize, numMaps[layerIndex], outputMapDim, outputMapDim))
layerIndex = layerIndex + 1

-- C4 layer (multi-scale) : 2 parallel branches
multiScaleLayer = nn.Concat(2) -- first dimension is is batch, 2nd is the feature

-- 1st branch : fully connected layer computing half of the feature
firstScaleBranch = nn.Sequential()
    -- change the dimensions from: depthXheightXwidthXbatch to BatchXdepthXheightXwidth
firstScaleBranch:add(nn.Transpose({4,1},{4,2},{4,3}))
local outputSize = numMaps[layerIndex - 1]*outputMapDim*outputMapDim
firstScaleBranch:add(nn.Reshape(outputSize, true))
firstScaleBranch:add(nn.Linear(outputSize, featureDim / 2))
multiScaleLayer:add(firstScaleBranch)

-- 2nd branch : locally connected layer followed by fully connected layer computing 2nd half of the feature
secondScaleBranch =  nn.Sequential()
inputDim = numMaps[layerIndex - 1]
inputMapDim = outputMapDim
secondScaleBranch:add(ccn2.SpatialConvolutionLocal(inputDim, numMaps[layerIndex], inputMapDim, filtersSize[layerIndex]))
secondScaleBranch:add(nn.ReLU())
outputMapDim = inputMapDim - filtersSize[layerIndex] + 1
print(string.format('C4 : %dx%dx%dx%d@%dx%d',
    numMaps[layerIndex], filtersSize[layerIndex], filtersSize[layerIndex], inputDim, outputMapDim, outputMapDim))
layerIndex = layerIndex + 1

    -- change the dimensions from: depthXheightXwidthXbatch to BatchXdepthXheightXwidth
secondScaleBranch:add(nn.Transpose({4,1},{4,2},{4,3}))
local outputSize = numMaps[layerIndex - 1]*outputMapDim*outputMapDim
secondScaleBranch:add(nn.Reshape(outputSize, true))
secondScaleBranch:add(nn.Linear(outputSize, featureDim / 2))
multiScaleLayer:add(secondScaleBranch)

model:add(multiScaleLayer)

-- Final layer F6 - classification into class out of nLabels classses
model:add(nn.Linear(featureDim, nLabels))

print '==> here is the model:'
print(model)

----------------------------------------------------------------------
print '==> initalizing weights'
for _, layerId in pairs(layersIds) do
    model:get(layerId).weight:normal(0, 0.01)
    model:get(layerId).bias:fill(0.5)
end
for iMultiScaleBranch = 1,2 do
    local multiScaleLayerIds = multiScaleTrainableLayerIds[iMultiScaleBranch]
    for iTrainableLayer = 1,#multiScaleLayerIds do
        local layerId = multiScaleLayerIds[iTrainableLayer]
        model:get(multiScaleLayerId):get(iMultiScaleBranch):get(layerId).weight:normal(0, 0.01)
        model:get(multiScaleLayerId):get(iMultiScaleBranch):get(layerId).bias:fill(0.5)
    end
end

if opt.visualize then
    require 'gfx.js'
    print '==> visualizing filters'
    for layerName, layerId in pairs(layersIds) do
        weights = model:get(layerId).weight
        print(string.format('%s : %dx%d', layerName, weights:size()[1], weights:size()[2]))
        gfx.image(weights, {zoom=2, legend=layerName})
    end
end