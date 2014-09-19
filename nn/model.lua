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
-- number of output maps for layers C1 
numMaps = {32, 16, 16, 16, 16}

-- TODO: ask about cuda fully connected layer

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
inputDim = 25 -- inputDim = (55 - 7) / 2 + 1
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
for _, layerId in ipairs({2,5,7,9,11,15,18}) do
    model:get(layerId).weight:normal(0, 0.01)
end


if opt.visualize then
    print '==> visualizing filters'
    gfx.image(model:get(2).weight, {zoom=2, legend='C1'})
    gfx.image(model:get(5).weight, {zoom=2, legend='C3'})
    gfx.image(model:get(7).weight, {zoom=2, legend='L4'})
    gfx.image(model:get(9).weight, {zoom=2, legend='L5'})
    gfx.image(model:get(11).weight, {zoom=2, legend='L6'})
    gfx.image(model:get(15).weight, {zoom=2, legend='F7'})
    gfx.image(model:get(18).weight, {zoom=2, legend='F8'})
end