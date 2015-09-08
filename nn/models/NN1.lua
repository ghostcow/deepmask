require 'nn'
require 'cunn'
require 'cudnn'

-- Same as scratch training directly with pruning.

print '==> building model'

local model = nn.Sequential()

-- Conv11 & Conv12 layers
model:add(cudnn.SpatialConvolution(1, 32, 3, 3, 1, 1, 1, 1)) --1
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialConvolution(32, 64, 3, 3, 1, 1, 1, 1)) --3
model:add(cudnn.ReLU(true))

-- Pool1
model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))

-- Conv21 & Conv22 layers
model:add(cudnn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1)) -- 6
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1)) -- 8
model:add(cudnn.ReLU(true))

-- Pool2
model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))

-- Conv31 & Conv 32 layers
model:add(cudnn.SpatialConvolution(128, 96, 3, 3, 1, 1, 1, 1)) -- 11
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialConvolution(96, 192, 3, 3, 1, 1, 1, 1)) -- 13
model:add(cudnn.ReLU(true))

-- Pool3
model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))

-- Conv41 & Conv42
model:add(cudnn.SpatialConvolution(192, 128, 3, 3, 1, 1, 1, 1)) -- 16
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1)) -- 18
model:add(cudnn.ReLU(true))

-- Pool 4
model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))

-- Conv51 & Conv52
model:add(cudnn.SpatialConvolution(256, 160, 3, 3, 1, 1, 1, 1)) -- 21
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

print('=> Initializing weights according to PReLU')
for i=1,#model.modules do
    local layer = model:get(i)
    if layer.weight ~= nil and layer.kW ~= nil then
        local stdv = math.sqrt(2/(1.25*layer.kW*layer.kH*layer.nInputPlane))
        layer.weight:normal(0, stdv)
    end
    if layer.bias ~= nil then
        layer.bias:fill(0)
    end
end

return model
