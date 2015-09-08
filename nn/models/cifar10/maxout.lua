require 'nn'
require 'cunn'


function createModel()
    print '==> building model'

    local model = nn.Sequential()

    -- Maxout1
    model:add(nn.Dropout(0.8))
    model:add(nn.SpatialConvolution(3, 96*2, 8, 8, 1, 1, 4, 4))
    model:add(nn.View(-1, 96*2, 33 * 33))
    model:add(nn.SpatialMaxPooling(1, 2, 1, 2))
    model:add(nn.View(-1, 96, 33, 33))

    -- Pool1
    model:add(nn.SpatialMaxPooling(4, 4, 2, 2))

    -- Maxout2
    model:add(nn.Dropout(0.5))
    model:add(nn.SpatialConvolution(96, 192*2, 8, 8, 1, 1, 3, 3))
    model:add(nn.View(-1, 192*2, 14 * 14))
    model:add(nn.SpatialMaxPooling(1, 2, 1, 2))
    model:add(nn.View(-1, 192, 14, 14))

    -- Pool2
    model:add(nn.SpatialMaxPooling(4, 4, 2, 2))

    -- Maxout3
    model:add(nn.Dropout(0.5))
    model:add(nn.SpatialConvolution(192, 192*2, 5, 5, 1, 1,3,3))
    model:add(nn.View(-1, 192*2, 8 * 8))
    model:add(nn.SpatialMaxPooling(1, 2, 1, 2))
    model:add(nn.View(-1, 192, 8, 8))

    -- Pool3
    model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

    -- Linear Maxout
    model:add(nn.Dropout(0.5))
    model:add(nn.View(-1, 192*4*4))
    model:add(nn.Linear(192*4*4, 500*5))
    model:add(nn.View(-1, 5, 500))
    model:add(nn.TemporalMaxPooling(5))
    model:add(nn.View(-1, 500))

    -- SoftMax
    model:add(nn.Dropout(0.5))
    model:add(nn.Linear(500, 10))
    model:add(nn.LogSoftMax())

    return model
end