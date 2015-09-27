require 'nn'
require 'cunn'
require 'cudnn'
require 'stn'

function build_st(sampleSize)
    local spanet = nn.Sequential()

    local concat = nn.ConcatTable()

    -- first branch is there to transpose inputs to BHWD, for the bilinear sampler
    local tranet = nn.Sequential()
    tranet:add(nn.Identity())
    tranet:add(nn.Transpose({2,3},{3,4}))

    -- second branch is the localization network
    local locnet = nn.Sequential()
    locnet:add(cudnn.SpatialMaxPooling(2,2,2,2))
    locnet:add(cudnn.SpatialConvolution(sampleSize[1],20,5,5))
    locnet:add(cudnn.ReLU(true))
    locnet:add(cudnn.SpatialMaxPooling(2,2,2,2))
    locnet:add(cudnn.SpatialConvolution(20,20,5,5))
    locnet:add(cudnn.ReLU(true))
    locnet:add(nn.View(20*19*19))
    locnet:add(nn.Linear(20*19*19,20))
    locnet:add(cudnn.ReLU(true))

    -- we initialize the output layer so it gives the identity transform
    local outLayer = nn.Linear(20,6)
    outLayer.weight:fill(0)
    local bias = torch.FloatTensor(6):fill(0)
    bias[1]=1
    bias[5]=1
    outLayer.bias:copy(bias)
    locnet:add(outLayer)

    -- there we generate the grids
    locnet:add(nn.View(2,3))
    locnet:add(nn.AffineGridGeneratorBHWD(sampleSize[2],sampleSize[3]))

    -- we need a table input for the bilinear sampler, so we use concattable
    concat:add(tranet)
    concat:add(locnet)

    spanet:add(concat)
    spanet:add(nn.BilinearSamplerBHWD())

    -- and we transpose back to standard BDHW format for subsequent processing by nn modules
    spanet:add(nn.Transpose({3,4},{2,3}))

    return spanet
end

-- Same as scratch training directly with pruning.
print '==> building model'

local model = nn.Sequential()

-- Add spatial image transform
model:add(build_st({1,100,100}))

-- Conv11 & Conv12 layers
model:add(cudnn.SpatialConvolution(1, 32, 3, 3, 1, 1, 1, 1)) --1
model:add(nn.SpatialBatchNormalization(32))
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialConvolution(32, 64, 3, 3, 1, 1, 1, 1)) --3
model:add(nn.SpatialBatchNormalization(64))
model:add(cudnn.ReLU(true))

-- Pool1
model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))

-- Conv21 & Conv22 layers
model:add(cudnn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1)) -- 6
model:add(nn.SpatialBatchNormalization(64))
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1)) -- 8
model:add(nn.SpatialBatchNormalization(128))
model:add(cudnn.ReLU(true))

-- Pool2
model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))

-- Conv31 & Conv 32 layers
model:add(cudnn.SpatialConvolution(128, 96, 3, 3, 1, 1, 1, 1)) -- 11
model:add(nn.SpatialBatchNormalization(96))
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialConvolution(96, 192, 3, 3, 1, 1, 1, 1)) -- 13
model:add(nn.SpatialBatchNormalization(192))
model:add(cudnn.ReLU(true))

-- Pool3
model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))

-- Conv41 & Conv42
model:add(cudnn.SpatialConvolution(192, 128, 3, 3, 1, 1, 1, 1)) -- 16
model:add(nn.SpatialBatchNormalization(128))
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1)) -- 18
model:add(nn.SpatialBatchNormalization(256))
model:add(cudnn.ReLU(true))

-- Pool 4
model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))

-- Conv51 & Conv52
model:add(cudnn.SpatialConvolution(256, 160, 3, 3, 1, 1, 1, 1)) -- 21
model:add(nn.SpatialBatchNormalization(160))
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialConvolution(160, 320, 3, 3, 1, 1, 1, 1)) -- 23

-- Pool 5
model:add(cudnn.SpatialAveragePooling(6, 6, 1, 1))

-- Dropout
model:add(nn.Dropout(0.4))

-- Fc6
model:add(nn.Reshape(320,true))
model:add(nn.Linear(320, #dataset.classes))
model:add(nn.LogSoftMax())

return model
