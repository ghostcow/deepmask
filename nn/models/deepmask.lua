require 'image'   -- for image transforms
require 'nn'      -- provides all sorts of trainable modules/layers
require 'nnx'
require 'cunn'
require 'cudnn'
require 'MyView'
----------------------------------------------------------------------
print '==> construct model'

-- DON'T FORGET TO CALL :TRAINING() AND :EVALUATING() WHEN TRAINING/EVALING FOR DROPOUT
-- vgg pretrained net
local trunk = torch.load('vgg_model_d_e/vgg16_deepmask.t7')
-- mask predictor, 56x56
-- only works for 4d tensors (bcz of transpose layer)
-- so for single image reshape to batchsize 1
-- TODO: write cuda implementation for bilinear upsampling layer
local mask = nn.Sequential()
mask:add(trunk)
mask:add(cudnn.SpatialConvolution(512,512,1,1))
mask:add(cudnn.ReLU(true))
mask:add(cudnn.SpatialConvolution(512,512,14,14))
mask:add(cudnn.SpatialConvolution(512,56*56,1,1))
mask:add(cudnn.ReLU(true))
mask:add(nn.View(56*56,-1):setNumInputDims(3))
mask:add(nn.Transpose{2,3})
-- at this point the channels represent the # of 224x224 patches in the image (=1 during training)
mask:add(nn.MyView(56,56))
mask:add(nn.SpatialUpSamplingNearest(4))
mask:add(nn.Sigmoid(true))

-- this was instead of nn.SpatialUpSamplingNearest(4)
-- TODO: put in cuda implementation of this bilinear upsampling layer
--mask:add(nn.Type('torch.FloatTensor'))
--mask:add(nn.SpatialReSampling{oheight=224, owidth=224})

-- object classifier
local score = nn.Sequential()
score:add(trunk)
score:add(cudnn.SpatialMaxPooling(2,2,2,2))
score:add(cudnn.SpatialConvolution(512,512,7,7))
score:add(cudnn.ReLU(true))
score:add(nn.Dropout(0.5))
score:add(cudnn.SpatialConvolution(512,1024,1,1))
score:add(cudnn.ReLU(true))
score:add(nn.Dropout(0.5))
score:add(cudnn.SpatialConvolution(1024,81,1,1))
score:add(nn.View(-1):setNumInputDims(3))

--[[
-- test net (with interleave trick)
-- move features around in conv order for interleave layer
ct = nn.ConcatTable()
for dy = 0,1 do
  for dx = 0,1 do
    ct:add(nn.SpatialZeroPadding(-dx,dx-1,-dy,dy-1))
  end
end

pt = nn.ParallelTable()
pt:add(classifier:clone('bias', 'weight'))
pt:add(classifier:clone('bias', 'weight'))
pt:add(classifier:clone('bias', 'weight'))
pt:add(classifier:clone('bias', 'weight'))

test_classifier = nn.Sequential()
test_classifier:add(ct)
test_classifier:add(pt)
test_classifier:add(nn.InterleaveTable())

test_branches = nn.ConcatTable()
test_branches:add(mask)
test_branches:add(test_classifier)

-- outputs two types
test_model = nn.Sequential()
test_model:add(nn.AlignmentLayer(16,2))
test_model:add(vgg)
test_model:add(test_branches)
]]

return mask, score
