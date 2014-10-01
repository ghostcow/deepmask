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
require 'options'

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
    print '==> processing options'
    opt = getOptions()
end

----------------------------------------------------------------------
-- defining output paths
os.execute('mkdir -p ' .. opt.save)
local state_file_path = paths.concat(opt.save, 'model.net')
local train_log_path = paths.concat(opt.save, 'train.log')
local test_log_path = paths.concat(opt.save, 'test.log')

-- Log results to files
trainLogger = optim.Logger(train_log_path)
testLogger = optim.Logger(test_log_path)
----------------------------------------------------------------------
if not opt.loadState then
    print '==> changing model to CUDA'
    model:add(nn.LogSoftMax())
    model:cuda()

    print '==> configuring optimizer - SGD'
    -- TODO: need to figure out all of this parameters - see training in deepface article
    optimMethod = optim.sgd
    optimState = {
        learningRate = opt.learningRate,
        weightDecay = opt.weightDecay,
        momentum = opt.momentum,
        learningRateDecay = opt.learningRateDecay -- TODO: "changed manually" at deepface training
    }

    print '==> defining cost function to NLL'
    criterion = nn.ClassNLLCriterion()
    criterion:cuda()
else
    print '==> loadind pre-trained net & optimization parameters'
    local x = torch.load(state_file_path)
    model = x.model
    optimState = x.optimState
    optimMethod = x.optimMethod
    criterion = x.criterion

    -- TODO : print loaded state details here
end

----------------------------------------------------------------------
print '==> defining some tools'
-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector
parameters,gradParameters = model:getParameters()

----------------------------------------------------------------------
print '==> defining training procedure'

function train()

-- epoch tracker
epoch = epoch or 1

-- local vars
local time = sys.clock()

-- do one epoch
print('==> doing epoch on training data:')
print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
local totalSize = 0
local trainDataChunk
for iChunk = 1,trainData.numChunks do
	trainDataChunk = trainData.getChunk(iChunk)
	local totalSize = totalSize + trainDataChunk:size()
	-- shuffle at each epoch
	shuffle = torch.randperm(trainDataChunk:size())
	for t = 1,trainDataChunk:size(),opt.batchSize do
	    -- disp progress
	    xlua.progress(t, trainDataChunk:size())

	    -- create mini batch
	    local inputs = torch.Tensor(opt.batchSize, 3, imageDim, imageDim)
	    local targets = torch.Tensor(opt.batchSize)
	    if ((t+opt.batchSize-1) > trainDataChunk:size()) then
	      -- we don't use the last samples
	      break
	    end
	    for i = t,(t+opt.batchSize-1) do
		 inputs[{i-t+1}] = trainDataChunk.data[shuffle[i]]
		 targets[{i-t+1}] = trainDataChunk.labels[shuffle[i]]
	    end
	    -- NOTE: we suppport training on CUDA only
	    inputs = inputs:cuda()

	    -- create closure to evaluate f(X) and df/dX
	    local feval = function(x)
		-- get new parameters
		if x ~= parameters then
		 parameters:copy(x)
		end

		-- reset gradients
		gradParameters:zero()

		-- evaluate function for complete mini batch
		-- estimate f
		local output = model:forward(inputs)
		numInputs = inputs:size()[1]
		local err = criterion:forward(output, targets)
		
		-- f is the average of all criterions
		local f = err
		
		-- estimate df/dW
		local df_do = criterion:backward(output, targets)
		model:backward(inputs, df_do)
		
		-- update confusion
		for i=1,numInputs do
		  confusion:add(output[i], targets[i])
		end

		-- normalize gradients and f(X)
		gradParameters:div(numInputs)

		-- return f and df/dX
		return f,gradParameters
	    end

	    optimMethod(feval, parameters, optimState)
	end
end

 -- time taken
 time = sys.clock() - time
 time = time / totalSize
 print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

 -- print confusion matrix
 print(confusion)
 local filename_confusion = paths.concat(opt.save, 'confusion_train')
 torch.save(filename_confusion, confusion)
 
 -- update logger/plot
 trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
 if opt.plot then
     trainLogger:style{['% mean class accuracy (train set)'] = '-'}
     trainLogger:plot()
 end

 -- save/log current net
 print('==> saving model & state to '..state_file_path)
 torch.save(state_file_path, {model=model, optimState=optimState, optimMethod=optimMethod, criterion = criterion})

 -- next epoch
 confusion:zero()
 epoch = epoch + 1
end
