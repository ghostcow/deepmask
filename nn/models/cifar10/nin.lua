require 'nn'
require 'cunn'
require 'cudnn'

function createModel()
    print '==> building model'

    local model = nn.Sequential()

    model:add(cudnn.SpatialConvolution(3, 192, 5, 5, 1, 1, 2, 2, 1))
    model:add(cudnn.ReLU(true))
    model:add(cudnn.SpatialConvolution(192, 160, 1, 1, 1, 1, 0, 0, 1))
    model:add(cudnn.ReLU(true))
    model:add(cudnn.SpatialConvolution(160, 96, 1, 1, 1, 1, 0, 0, 1))
    model:add(cudnn.ReLU(true))
    model:add(cudnn.SpatialMaxPooling(3, 3, 2, 2):ceil())
    model:add(nn.Dropout(0.500000))
    model:add(cudnn.SpatialConvolution(96, 192, 5, 5, 1, 1, 2, 2, 1))
    model:add(cudnn.ReLU(true))
    model:add(cudnn.SpatialConvolution(192, 192, 1, 1, 1, 1, 0, 0, 1))
    model:add(cudnn.ReLU(true))
    model:add(cudnn.SpatialConvolution(192, 192, 1, 1, 1, 1, 0, 0, 1))
    model:add(cudnn.ReLU(true))
    model:add(cudnn.SpatialAveragePooling(3, 3, 2, 2):ceil())
    model:add(nn.Dropout(0.500000))
    model:add(cudnn.SpatialConvolution(192, 192, 3, 3, 1, 1, 1, 1, 1))
    model:add(cudnn.ReLU(true))
    model:add(cudnn.SpatialConvolution(192, 192, 1, 1, 1, 1, 0, 0, 1))
    model:add(cudnn.ReLU(true))
    model:add(cudnn.SpatialConvolution(192, 10, 1, 1, 1, 1, 0, 0, 1))
    model:add(cudnn.ReLU(true))
    model:add(cudnn.SpatialAveragePooling(8, 8, 1, 1):ceil())
    model:add(nn.SoftMax())

    return model
end