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
require 'deep_id_utils'
require 'math'

-- extract patch scale
iPatch,iScale,iType = DeepIdUtils.parsePatchIndex(opt.patchIndex)
imageDim = DeepIdUtils.patchSizeTarget[(iType - 1)*DeepIdUtils.numScales + iScale]
if (type(imageDim) == 'table') then
    imageDim = imageDim[1]
end

-- only rgb patches are supported for now
local inputDim = 3
-- deafult value for feature dimension
featureDim = 160

-- filter sizes & number of maps for layers C1,C2,C3,C4
-- these sizes changes according to input dimension
-- the output of M3 (for model subModelType='3') is fixed to 32x2x2
if (imageDim == 31) or (imageDim == 39) or (imageDim == 47) then
    filtersSize = {4, 3, 3, 2}
elseif (imageDim == 45) or (imageDim == 53) or (imageDim == 61) then
    filtersSize = {6, 5, 5, 4}
elseif (imageDim == 59) or (imageDim == 67) or (imageDim == 75) then
    filtersSize = {8, 7, 7, 6}
end

--numMaps = {32, 48, 64, 80} -- NOTE: original paper uses {20, 40, 60, 80}
--- NOTE : due to stablity issues (explosion of gradients) we use these maps & deepID.3.64 / deepID.3.160
numMaps = {16, 32, 32, 48} -- NOTE: original paper uses {20, 40, 60, 80}
-- also works : numMaps = {16, 16, 16, 16} & deepID.full.64

maxPoolingSize = 2
maxPoolingStride = 2

-- layers ids (except the multi-scale layer)
layersIds = {C1=2,C2=5,C3=8,F6=15}
multiScaleLayerId = 11
-- F5 layer is divied into 2 parts in the multi scale layer
multiScaleTrainableLayerIds = {{3}, {1,5} }

----------------------------------------------------------------------
-- use -visualize to show network
-- parse command line arguments
if not opt then
    print '==> processing options'
    opt = getOptions()
end
k = string.find(opt.modelName, '.', 1, true)
if k then
    subModelType = opt.modelName:sub(k+1) --- 3 = cutting the multi-scale layer (after M3)
    k = string.find(subModelType, '.', 1, true)
    if k then
        featureDim = tonumber(subModelType:sub(k+1))
        subModelType = subModelType:sub(1, k-1)

        print(subModelType)
        print(featureDim)
    end
else
    subModelType = 'full'
end
if (subModelType == '3') then
    layersIds['F4'] = 13
    layersIds['F5'] = 15
    --- delete F6
    layersIds['F6'] = nil

    multiScaleLayerId = nil
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
    outputMapDim = math.floor(inputMapDim / maxPoolingStride)
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
outputMapDim = math.floor(inputMapDim / maxPoolingStride)
print(string.format('M3 : %dx%dx%dx%d@%dx%d',
    numMaps[layerIndex], maxPoolingSize, maxPoolingSize, numMaps[layerIndex], outputMapDim, outputMapDim))
layerIndex = layerIndex + 1

-- C4 layer (multi-scale) : 2 parallel branches
if (subModelType == 'full') then
    multiScaleLayer = nn.Concat(2) -- first dimension is is batch, 2nd is the feature

    -- 1st branch : fully connected layer computing half of the feature
    firstScaleBranch = nn.Sequential()
        -- change the dimensions from: depthXheightXwidthXbatch to BatchXdepthXheightXwidth
    firstScaleBranch:add(nn.Transpose({4,1},{4,2},{4,3}))
    local outputSize = numMaps[layerIndex - 1]*outputMapDim*outputMapDim
    firstScaleBranch:add(nn.Reshape(outputSize, true))
    --firstScaleBranch:add(nn.View(-1):setNumInputDims(2))
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
    -- secondScaleBranch:add(nn.View(-1):setNumInputDims(2))
    secondScaleBranch:add(nn.Linear(outputSize, featureDim / 2))
    multiScaleLayer:add(secondScaleBranch)

    model:add(multiScaleLayer)
elseif (subModelType == '3') then
    model:add(nn.Transpose({4,1},{4,2},{4,3}))
    local outputSize = numMaps[layerIndex - 1]*outputMapDim*outputMapDim
    -- model:add(nn.Reshape(outputSize, true))
    model:add(nn.View(-1, outputSize):setNumInputDims(2))
    model:add(nn.Linear(outputSize, featureDim))
elseif (subModelType == '1scale') then
    inputDim = numMaps[layerIndex - 1]
    inputMapDim = outputMapDim
    model:add(ccn2.SpatialConvolutionLocal(inputDim, numMaps[layerIndex], inputMapDim, filtersSize[layerIndex]))
    model:add(nn.ReLU())
    outputMapDim = inputMapDim - filtersSize[layerIndex] + 1
    print(string.format('C4 : %dx%dx%dx%d@%dx%d',
        numMaps[layerIndex], filtersSize[layerIndex], filtersSize[layerIndex], inputDim, outputMapDim, outputMapDim))
    layerIndex = layerIndex + 1

    -- change the dimensions from: depthXheightXwidthXbatch to BatchXdepthXheightXwidth
    model:add(nn.Transpose({4,1},{4,2},{4,3}))
    local outputSize = numMaps[layerIndex - 1]*outputMapDim*outputMapDim
    model:add(nn.View(-1, outputSize):setNumInputDims(2))
    model:add(nn.Linear(outputSize, featureDim))
end
-- model:add(nn.ReLU()) --activation function for the previous fully-connected layer

-- Final layer F6 - classification into class out of nLabels classses
model:add(nn.Dropout())
model:add(nn.Linear(featureDim, nLabels))

print '==> here is the model:'
print(model)

----------------------------------------------------------------------
if opt.visualize then
    require 'gfx.js'
    print '==> visualizing filters'
    for layerName, layerId in pairs(layersIds) do
        weights = model:get(layerId).weight
        print(string.format('%s : %dx%d', layerName, weights:size()[1], weights:size()[2]))
        gfx.image(weights, {zoom=2, legend=layerName})
    end
end