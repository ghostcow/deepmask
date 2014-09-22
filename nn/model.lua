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
require 'gfx.js'

imagedim = 152
-- filter sizes for layers C1,C3,L4,L5,L6
filtersSize = {11, 9, 9, 7, 5}
maxPoolingStride = 2
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
    cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Deepface torch7 model')
    cmd:text()
    cmd:text('Options:')
    cmd:option('-visualize', false, 'visualize input data and weights during training')
    cmd:text()
    opt = cmd:parse(arg or {})
end

----------------------------------------------------------------------
print '==> building model'

model = nn.Sequential()

-- convert from Torch batchSizeX3XheightXwidth format to ccn2 format 3XheightXwidthXbatchSize
model:add(nn.Transpose({1,4},{1,3},{1,2}))

-- Map construction format
-- ccn2.SpatialConvolution(inputDim, #feature-maps, filter-size)
-- ccn2.SpatialMaxPooling(neighberhoodSize, stride)
-- ccn2.SpatialConvolutionLocal(inputDim, #feature-maps, inputDim, filter-size)

-- C1 layer
model:add(ccn2.SpatialConvolution(3, numMaps[1], filtersSize[1])) -- 1
model:add(nn.ReLU())
outputMapDim = imagedim - filtersSize[1] + 1

-- M2 layer
model:add(ccn2.SpatialMaxPooling(3,maxPoolingStride)) -- 3
outputMapDim = outputMapDim / maxPoolingStride

-- C3 layer
model:add(ccn2.SpatialConvolution(numMaps[1], numMaps[2], filtersSize[2])) -- 4
model:add(nn.ReLU())
outputMapDim = outputMapDim - filtersSize[2] + 1

-- L4 layer
inputDim = outputMapDim
model:add(ccn2.SpatialConvolutionLocal(numMaps[2], numMaps[3], inputDim, filtersSize[3])) -- 6
model:add(nn.ReLU())
outputMapDim = outputMapDim - filtersSize[3] + 1

-- L5 layer
model:add(ccn2.SpatialConvolutionLocal(numMaps[3], numMaps[4], outputMapDim, filtersSize[4], L5_stride)) -- 8
model:add(nn.ReLU())

-- L6 layer
inputDim = (outputMapDim - filtersSize[4])/L5_stride + 1
model:add(ccn2.SpatialConvolutionLocal(numMaps[4], numMaps[5], inputDim, filtersSize[5])) -- 10
model:add(nn.ReLU())
outputDim = inputDim - filtersSize[5] + 1

-- change the dimensions from: depthXheightXwidthXbatch to BatchXdepthXheightXwidth
model:add(nn.Transpose({4,1},{4,2},{4,3}))
-- transform the output into a vector
model:add(nn.Reshape(numMaps[4]*outputDim*outputDim, true))


-- F7 layer
model:add(nn.Linear(numMaps[4]*outputDim*outputDim, 4096)) -- 14
model:add(nn.ReLU())
model:add(nn.Dropout())

-- F8 layer - classification into class out of numPersons classses
model:add(nn.Linear(4096, numPersons)) -- 17


print '==> here is the model:'
print(model)

----------------------------------------------------------------------
print '==> initalizing weights'
for _, layerId in pairs(layersIds) do
    model:get(layerId).weight:normal(0, 0.01)
    model:get(layerId).bias:fill(0.5)
end


if opt.visualize then
    print '==> visualizing filters'
    for layerName, layerId in pairs(layersIds) do
        gfx.image(model:get(layerId).weight, {zoom=2, legend=layerName})
    end
end
