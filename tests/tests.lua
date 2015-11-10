local function myview_test()
  require 'nn'
  require 'toolbox/MyView'

  -- make
  mv = nn.MyView(3,3)
  t = torch.randn(3,9)
  print(t)
  a = mv:forward(t)
  print(a)
  b = mv:backward(t, torch.randn(3,3,3))
  print(b)
  -- break
  --  mv = nn.MyView(3,4)
  --  t = torch.randn(3,9)
  --  print(t)
  --  a = mv:forward(t)
  --  print(a)
  --  b = mv:backward(t, torch.randn(3,3,3))
  --  print(b)

end

local function resample_test()
  require 'image'
  require 'nnx'
  img = image.loadPNG('/home/lioruzan/Desktop/esty.png')
  imsz = img:size()
  input = img:view(1,imsz[1],imsz[2],imsz[3]):repeatTensor(3,1,1,1)
  l = nn.SpatialReSampling{rwidth=2,rheight=2}
  output = l:forward(input)
  print(output:size())
  --  image.save('esty-x2-bilinear.png', output)
  dfdx = l:backward(input, torch.rand(output:size()))
  print(dfdx:size())
end

local function view_test()
  require 'nn'
  local batchSize = 2
  view = nn.View(batchSize, 784, -1)
  t1 = torch.Tensor(batchSize, 784, 2,2)
  t2 = view:forward(t1)
  print(t1:size())
  print(t2:size())
end

local function interleave_test()
  require 'nn'
  require 'toolbox/TableInterleave'
  local ti = nn.TableInterleave()
  print('-- batch size = 1, input channels = 1')
  local input = {}
  for i=1,4 do table.insert(input, torch.Tensor(3,3):fill(i)) end
  local grad = torch.Tensor(6,6):fill(-1)
  print(ti:forward(input))
  print(ti:backward(input,grad))
  print('-- batch size = 1, input channels > 1')
  input = {}
  for i=1,4 do table.insert(input, torch.Tensor(1,3,3,3):fill(i)) end
  local grad = torch.Tensor(1,3,6,6):fill(-1)
  print(ti:forward(input))
  print(ti:backward(input,grad))
  print('-- batch size > 1, input channels > 1')
  input = {}
  for i=1,4 do table.insert(input, torch.Tensor(4,3,3,3):fill(i)) end
  local grad = torch.Tensor(4,3,6,6):fill(-1)
  print(ti:forward(input))
  print(ti:backward(input,grad))
end

local function alignmentlayer_test()
  -- testing layer
  al = nn.AlignmentLayer(16,2):float()
  -- non batch
  input = torch.FloatTensor(3,512,512)
  fw = al:forward(input)
  bk = al:backward(input, fw:clone())
  print(fw:size(), bk:size())
  -- batch
  input = torch.FloatTensor(2,3,512,513)
  fw = al:forward(input)
  bk = al:backward(input, fw:clone())
  print(fw:size(), bk:size())
end

local function train_net_test()
    dofile('net/2_model.lua')
    collectgarbage()
    -- cuda net
    model:cuda()
    -- mask.modules[10].output = mask.modules[10].output:float()
    mask.modules[11]:float()
    -- train network test

    -- single
    local input = image.lena(); input = image.scale(input,224,224):resize(1,3,224,224):cuda()
    local fw = time_func('single fw time: ', model.forward, model, input)
    local bk = time_func('single bk time: ', model.backward, model, input, {torch.randn(1,1,224,224), torch.randn(1,1,1,1):cuda()})

    -- batch

    local bsize = 32
    input = input:resize(3,224,224):float(); input = (function() local t={}; for i=1,bsize do table.insert(t,input:clone()) end; return torch.vstack(t) end)()
    input = input:cuda()
    fw = time_func('batch fw time: ', model.forward, model, input)
    bk = time_func('batch bk time: ', model.backward, model, input, {torch.randn(bsize,1,224,224), torch.randn(bsize,1,1,1):cuda()})
    -- print(bk:size())

    -- test network test
    -- input = image.lena()
end

-- call after train_net_test or something
local function test_net_test()
    dofile('net/2_model.lua')
    collectgarbage()
    test_model:cuda()
    test_branches.modules[1].modules[11]:float()
    local dimsize = 224
    -- single pic
    local input = image.lena(); input = image.scale(input,dimsize,dimsize):resize(1,3,dimsize,dimsize):cuda()
    local fw = time_func('single fw time: ', test_model.forward, test_model, input)
    -- batch
    local bsize = 32
    local raw_input = image.scale(image.lena(), dimsize, dimsize)
    input = (function() local t={}; for i=1,bsize do table.insert(t,raw_input:clone()) end; return torch.vstack(t); end)()
    fw = time_func('batch size ' .. tostring(bsize) .. ' fw time: ', test_model.forward, test_model, input:cuda())
end

-- test layers
require 'sys'
sys.tic()
myview_test()
print('MyView test took ' .. sys.toc() .. ' seconds')
sys.tic()
resample_test()
print('Resample test took ' .. sys.toc() .. ' seconds')
sys.tic()
view_test()
print('View test took ' .. sys.toc() .. ' seconds')
sys.tic()
interleave_test()
print('TableInterleave took ' .. sys.toc() .. ' seconds')
sys.tic()
alignmentlayer_test()
print('View test took ' .. sys.toc() .. ' seconds')

-- test nets
-- TODO: complete this part