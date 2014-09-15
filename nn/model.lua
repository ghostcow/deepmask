----------------------------------------------------------------------
-- Deepface nn model for torch7
--
-- Input: 3X152X152X(batchsize)
-- Outout: 4030 vector (aka SFC ID)
----------------------------------------------------------------------
require 'nn'
require 'cunn'
require 'ccn2'
require 'torch'
require 'image'
require 'gfx.js'


-- TODO: ask about cuda fully connected layer
-- TODO: added Dropout layer with correct param

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

model = nn.Sequential()

-- C1 layer
model:add(ccn2.SpatialConvolution(3, 32, 11)) -- 2
model:add(nn.ReLU())

-- M2 layer
model:add(ccn2.SpatialMaxPooling(3,2)) -- 4

-- C3 layer
model:add(ccn2.SpatialConvolution(32, 16, 9)) --5
model:add(nn.ReLU())

-- L4 layer
model:add(ccn2.SpatialConvolutionLocal(16, 16, 63, 9)) -- 7
model:add(nn.ReLU())

-- L5 layer
model:add(ccn2.SpatialConvolutionLocal(16, 16, 55, 7, 2)) -- 9
model:add(nn.ReLU())

-- L6 layer
model:add(ccn2.SpatialConvolutionLocal(16, 16, 25, 5)) -- 11
model:add(nn.ReLU())


--TODO: needs to be in format 1X1XWIDTHXdepth (maybe sue transpose)
model:add(nn.Reshape(16*16*21)) -- transform the output into a vector


-- F7 layer
model:add(nn.Linear(16*16*21, 4096)) -- 14
model:add(nn.ReLU())
model:add(nn.Dropout())

-- F8 layer
model:add(nn.Linear(4096, 4030)) -- 17


print '==> here is the model:'
print(model)

if opt.visualize then
    print '==> visualizing filters'
    gfx.image(model:get(1).weight, {zoom=2, legend='C1'})
    gfx.image(model:get(4).weight, {zoom=2, legend='C3'})
    -- NOTE: gfx fails to visualize local layers, this bug is related to the filter size
    gfx.image(model:get(6).weight, {zoom=2, legend='L4'})
    gfx.image(model:get(8).weight, {zoom=2, legend='L5'})
    gfx.image(model:get(10).weight, {zoom=2, legend='L6'})
    gfx.image(model:get(13).weight, {zoom=2, legend='F7'})
    gfx.image(model:get(16).weight, {zoom=2, legend='F8'})
end