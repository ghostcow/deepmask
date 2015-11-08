require 'image'   -- for image transforms
require 'nn'      -- provides all sorts of trainable modules/layers
require 'cunn'
require 'cudnn'
require 'toolbox'

----------------------------------------------------------------------
print '==> construct model'

-- DON'T FORGET TO CALL :TRAINING() AND :EVALUATING() WHEN TRAINING/EVALING FOR DROPOUT
-- vgg pretrained net
local vgg = torch.load('vgg_model_d_e/vgg16_deepmask.t7')
-- mask predictor, 56x56
-- only works for 4d tensors (bcz of transpose layer)
-- so for single image reshape to batchsize 1
local mask = nn.Sequential()
mask:add(nn.SpatialConvolutionMM(512,512,1,1,1,1))
mask:add(nn.ReLU(true))
mask:add(nn.SpatialConvolutionMM(512,512,14,14,1,1))
mask:add(nn.ReLU(true))
mask:add(nn.SpatialConvolutionMM(512,56*56,1,1,1,1))
mask:add(nn.ReLU(true))
mask:add(nn.View(56*56,-1):setNumInputDims(3))
mask:add(nn.Transpose{2,3})
mask:add(nn.MyView(56,56))
mask:add(nn.Type('torch.FloatTensor'))
mask:add(nn.SpatialReSampling{oheight=224, owidth=224})
-- objectness classifier
local classifier = nn.Sequential()
classifier:add(nn.SpatialMaxPooling(2,2,2,2))
classifier:add(nn.SpatialConvolutionMM(512,512,7,7,1,1))
classifier:add(nn.ReLU(true))
classifier:add(nn.Dropout(0.5))
classifier:add(nn.SpatialConvolutionMM(512,1024,1,1,1,1))
classifier:add(nn.ReLU(true))
classifier:add(nn.Dropout(0.5))
classifier:add(nn.SpatialConvolutionMM(1024,1,1,1,1,1))

-- training net
local branches = nn.ConcatTable()
branches:add(mask)
branches:add(classifier)

local model = nn.Sequential()
model:add(vgg)
model:add(branches)

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
return model
