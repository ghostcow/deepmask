----------------------------------------------------------------------
-- Deepface nn train code torch7
--
-- Uses CUDA
----------------------------------------------------------------------

require 'torch'
require 'nn'
require 'torch'
require 'xlua'
require 'optim'

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
    print '==> processing options'
    cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Deepface Training/Optimization')
    cmd:text()
    cmd:text('Options:')
    cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
    cmd:option('-visualize', false, 'visualize input data and weights during training')
    cmd:option('-plot', false, 'live plot')
    cmd:option('-learningRate', 0.01, 'learning rate at t=0')
    cmd:option('-batchSize', 128, 'mini-batch size (1 = pure stochastic)')
    cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
    cmd:option('-momentum', 0.9, 'momentum (SGD only)')
    cmd:text()
    opt = cmd:parse(arg or {})
end

----------------------------------------------------------------------
print '==> defining cost function to nll, and changing model to CUDA'

model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()
model:cuda()
criterion:cuda()

----------------------------------------------------------------------
print '==> defining some tools'

-- classes
-- TODO: is this intersting for us?, If so, need to update classes
classes = {'1','2','3','4','5','6','7','8','9','0'}

-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- Log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector
if model then
 parameters,gradParameters = model:getParameters()
end

----------------------------------------------------------------------
print '==> configuring optimizer - SGD'
-- TODO: need to figure out all of this parameters - see training in deepface article
optimMethod = optim.sgd
optimState = {
    learningRate = opt.learningRate,
    weightDecay = opt.weightDecay,
    momentum = opt.momentum,
    learningRateDecay = 1e-7 -- TODO: "changed manually" at deepface training
}

-- TODO: initalize weights the same way as deepface
----------------------------------------------------------------------
print '==> defining training procedure'

function train()

-- epoch tracker
epoch = epoch or 1

-- local vars
local time = sys.clock()

-- shuffle at each epoch
shuffle = torch.randperm(trsize)

-- do one epoch
print('==> doing epoch on training data:')
print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
for t = 1,trainData:size(),opt.batchSize do
    -- disp progress
    xlua.progress(t, trainData:size())

    -- create mini batch
    local inputs = {}
    local targets = {}
    for i = t,math.min(t+opt.batchSize-1,trainData:size()) do
         -- load new sample
         local input = trainData.data[shuffle[i]]
         local target = trainData.labels[shuffle[i]]

         -- NOTE: we suppport training on CUDA only
         input = input:cuda()

         table.insert(inputs, input)
         table.insert(targets, target)
    end

    -- create closure to evaluate f(X) and df/dX
    local feval = function(x)
        -- get new parameters
        if x ~= parameters then
         parameters:copy(x)
        end

        -- reset gradients
        gradParameters:zero()

        -- f is the average of all criterions
        local f = 0

        -- evaluate function for complete mini batch
        for i = 1,#inputs do
         -- estimate f
         local output = model:forward(inputs[i])
         local err = criterion:forward(output, targets[i])
         f = f + err

         -- estimate df/dW
         local df_do = criterion:backward(output, targets[i])
         model:backward(inputs[i], df_do)

         -- update confusion
         confusion:add(output, targets[i])
        end

        -- normalize gradients and f(X)
        gradParameters:div(#inputs)
        f = f/#inputs

        -- return f and df/dX
        return f,gradParameters
    end

    optimMethod(feval, parameters, optimState)
end

 -- time taken
 time = sys.clock() - time
 time = time / trainData:size()
 print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

 -- print confusion matrix
 print(confusion)

 -- update logger/plot
 trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
 if opt.plot then
     trainLogger:style{['% mean class accuracy (train set)'] = '-'}
     trainLogger:plot()
 end

 -- save/log current net
 local filename = paths.concat(opt.save, 'model.net')
 os.execute('mkdir -p ' .. sys.dirname(filename))
 print('==> saving model to '..filename)
 torch.save(filename, model)

 -- next epoch
 confusion:zero()
 epoch = epoch + 1
end