require 'options'

----------------------------------------------------------------------
print '==> processing options'
opt = getOptions()
-- opt.visualize = true
opt.size = 'full'

-- nb of threads and fixed seed (for repeatable experiments)
torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)

----------------------------------------------------------------------
imageDim = 152
numPersons = 240

dofile 'model2.lua'

model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()
model:cuda()
criterion:cuda()

print 'flattening all parameters'
-- stack all trainable parameters
parameters,gradParameters = model:getParameters()

-- random input
print '--------generating random inputs--------'
inputs = torch.rand(opt.batchSize, 3, imageDim, imageDim)
inputs = inputs:cuda()
targets = torch.Tensor(1, opt.batchSize)
targets:fill(1)

-- forward & backward pass
print '--------forward pass--------'
output = model:forward(inputs)
t = targets[1]
err = criterion:forward(output, t)
print(string.foramt('error = %f', err))
print '--------backward pass--------'
df_do = criterion:backward(output, t)
model:backward(inputs, df_do)

print '--------successfull test!--------'