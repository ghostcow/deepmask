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
model:add(build_st({3,142,170}))

-- Conv11 & Conv12 & Pool1
model:add(cudnn.SpatialConvolution(3, 64, 3, 3, 1, 1, 1, 1)) --1
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1)) --3
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))

-- Conv21 & Conv22 & Pool2
model:add(cudnn.SpatialConvolution(128, 64, 3, 3, 1, 1, 1, 1)) -- 6
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1)) -- 8
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))

-- Conv31 & Conv32 & Pool 4
model:add(cudnn.SpatialConvolution(128, 64, 3, 3, 1, 1, 1, 1)) -- 11
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1)) -- 13
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))

-- Conv41 & Conv42 & Pool 4
model:add(cudnn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1)) -- 16
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1)) -- 18
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))

-- Conv51 & Conv52 & Pool5
model:add(cudnn.SpatialConvolution(256, 128, 3, 3, 1, 1, 1, 1)) -- 21
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1)) -- 23
model:add(cudnn.SpatialAveragePooling(2, 2, 2, 2))

-- Dropout & Fc6 & Fc7 & LogSoftMax
model:add(nn.View(256*4*5))
model:add(nn.Dropout(0.4))
model:add(nn.Linear(256*4*5, 512))
model:add(nn.Linear(512, #dataset.classes))
model:add(nn.LogSoftMax())

return model