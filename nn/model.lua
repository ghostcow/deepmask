----------------------------------------------------------------------
-- Deepface nn model for torch7
--
-- Input: (batchsize)X3X152X152
-- Outout: 4030 vector (aka SFC ID)
----------------------------------------------------------------------
require 'nn'
require 'cunn'
require 'ccn2'
require 'torch'
require 'image'
require 'options'

imageDim = 152
-- filter sizes for layers C1,C3,L4,L5,L6
filtersSize = {11, 9, 9, 7, 5}
maxPoolingStride = 2
maxPoolingSize = 3
L5_stride = 2
-- number of output maps for layers C1,C3,L4,L5,L6
numMaps = {32, 16, 16, 16, 16}
-- layers id
layersIds = {C1=2,C3=5,L4=7,L5=9,L6=11,F7=15,F8=18}

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

local inputDim = 3 -- number of input maps of any layer

local inputMapDim = imageDim -- dimension of input map to any layer
local outputMapDim 	     -- dimension of output map of any layer

-- Map construction format
-- ccn2.SpatialConvolution(inputDim, #feature-maps, filter-size)
-- ccn2.SpatialMaxPooling(neighberhoodSize, stride)
-- ccn2.SpatialConvolutionLocal(inputDim, #feature-maps, inputMapDim, filter-size)

-- C1 layer
local layerIndex = 1
model:add(ccn2.SpatialConvolution(inputDim, numMaps[layerIndex], filtersSize[layerIndex])) -- 1
model:add(nn.ReLU())
outputMapDim = inputMapDim - filtersSize[layerIndex] + 1
print(string.format('C1 : %dx%dx%dx%d@%dx%d', 
	numMaps[layerIndex], filtersSize[layerIndex], filtersSize[layerIndex], inputDim, outputMapDim, outputMapDim))

-- M2 layer
inputMapDim = outputMapDim
model:add(ccn2.SpatialMaxPooling(maxPoolingSize, maxPoolingStride)) -- 3
outputMapDim = inputMapDim / maxPoolingStride
print(string.format('M2 : %dx%dx%dx%d@%dx%d', 
	numMaps[layerIndex], maxPoolingSize, maxPoolingSize, numMaps[layerIndex], outputMapDim, outputMapDim))
layerIndex = layerIndex + 1

-- C3 layer
inputDim = numMaps[layerIndex - 1]
inputMapDim = outputMapDim
model:add(ccn2.SpatialConvolution(inputDim, numMaps[layerIndex], filtersSize[layerIndex])) -- 4
model:add(nn.ReLU())
outputMapDim = inputMapDim - filtersSize[layerIndex] + 1
print(string.format('C3 : %dx%dx%dx%d@%dx%d', 
	numMaps[layerIndex], filtersSize[layerIndex], filtersSize[layerIndex], inputDim, outputMapDim, outputMapDim))
layerIndex = layerIndex + 1

-- L4 layer
inputDim = numMaps[layerIndex - 1]
inputMapDim = outputMapDim
model:add(ccn2.SpatialConvolutionLocal(inputDim, numMaps[layerIndex], inputMapDim, filtersSize[layerIndex])) -- 6
model:add(nn.ReLU())
outputMapDim = inputMapDim - filtersSize[layerIndex] + 1
print(string.format('L4 : %dx%dx%dx%d@%dx%d', 
	numMaps[layerIndex], filtersSize[layerIndex], filtersSize[layerIndex], inputDim, outputMapDim, outputMapDim))
layerIndex = layerIndex + 1

-- L5 layer
inputDim = numMaps[layerIndex - 1]
inputMapDim = outputMapDim
model:add(ccn2.SpatialConvolutionLocal(inputDim, numMaps[layerIndex], inputMapDim, filtersSize[layerIndex], L5_stride)) -- 8
model:add(nn.ReLU())
outputMapDim = math.ceil((inputMapDim - filtersSize[layerIndex])/L5_stride) + 1
print(string.format('L5 : %dx%dx%dx%d@%dx%d', 
	numMaps[layerIndex], filtersSize[layerIndex], filtersSize[layerIndex], inputDim, outputMapDim, outputMapDim))
layerIndex = layerIndex + 1

-- L6 layer
inputDim = numMaps[layerIndex - 1]
inputMapDim = outputMapDim
model:add(ccn2.SpatialConvolutionLocal(inputDim, numMaps[layerIndex], inputMapDim, filtersSize[layerIndex])) -- 10
model:add(nn.ReLU())
outputMapDim = inputMapDim - filtersSize[layerIndex] + 1
print(string.format('L6 : %dx%dx%dx%d@%dx%d', 
	numMaps[layerIndex], filtersSize[layerIndex], filtersSize[layerIndex], inputDim, outputMapDim, outputMapDim))
layerIndex = layerIndex + 1

-- change the dimensions from: depthXheightXwidthXbatch to BatchXdepthXheightXwidth
model:add(nn.Transpose({4,1},{4,2},{4,3}))
-- transform the output into a vector
local outputSize = numMaps[layerIndex - 1]*outputMapDim*outputMapDim
model:add(nn.Reshape(outputSize, true))

-- F7 layer
model:add(nn.Linear(outputSize, 4096)) -- 14
model:add(nn.ReLU())
model:add(nn.Dropout())

-- F8 layer - classification into class out of numPersons classses
model:add(nn.Linear(4096, nLabels)) -- 17

print '==> here is the model:'
print(model)

----------------------------------------------------------------------
print '==> initalizing weights'
for _, layerId in pairs(layersIds) do
    model:get(layerId).weight:normal(0, 0.01)
    model:get(layerId).bias:fill(0.5)
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
