----------------------------------------------------------------------
-- Deepface nn train code torch7
--
-- Uses CUDA
----------------------------------------------------------------------

require 'torch'
require 'nn'
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
local state_file_path = paths.concat(opt.save, 'model.net')
local state_file_last_path = paths.concat(opt.save, 'model_last.net')
local train_log_path = paths.concat(opt.save, 'train.log')
local test_log_path = paths.concat(opt.save, 'test.log')

-- Log results to files
trainLogger = optim.Logger(train_log_path, true)
testLogger = optim.Logger(test_log_path, true)
timestamp = os.date("%Y_%m_%d_%X")
----------------------------------------------------------------------
optimState = {
           learningRate = opt.learningRate,
           weightDecay = opt.weightDecay,
           momentum = opt.momentum,
           learningRateDecay = opt.learningRateDecay
           }
print '==> configuring optimizer - SGD'           
optimMethod = optim.sgd
print '==> defining cost function to NLL'
if opt.balanceClasses then
    print '==> balance classes in NLL'

    --- compute frequency for each class
    class_freqs = torch.Tensor(nLabels):fill(0)
    for iChunk = 1,trainData.numChunks do
        trainDataChunk = trainData.getChunk(iChunk)
        for iImage = 1,trainDataChunk:size() do
            label = trainDataChunk.labels[iImage]
            class_freqs[label] = class_freqs[label] + 1
        end
    end
    print('classes frequencies :')
    print(class_freqs)
    class_weights = torch.Tensor(nLabels):fill(0)
    for iLabel = 1,nLabels do
        class_weights[iLabel] = 1 / class_freqs[iLabel]
    end
    class_weights = class_weights / class_weights:sum()
    print('classes weights :')
    print(class_weights)

    criterion = nn.ClassNLLCriterion(class_weights)
else
    criterion = nn.ClassNLLCriterion()
end
criterion:cuda()

if not opt.loadState then
    -- if model already exist, ask user before overwrite
    if os.rename(state_file_path, state_file_path) then --check if model file exist
        while true do
            io.stdout:write('Model file ', state_file_path, ' already exist. Sure you want to overwrite? [y/n]')
            ans = io.stdin:read()
            if (ans == 'y') then
                break
            elseif (ans == 'n') then
                os.exit()
            end
        end
    end

    print '==> changing model to CUDA'
    model:add(nn.LogSoftMax())
    model:cuda()
else
    print '==> loadind pre-trained net & optimization parameters'
    model = torch.load(state_file_path)
    
   -- read new learning rate from opt
   optimState.learningRate = opt.learningRate

   print '==> model :'
   print(model)
   print '==> optimState :'
   print(optimState)
   print '==> optimMethod :'
   print(optimMethod)
   print '==> criterion :'
   print(criterion)
end

----------------------------------------------------------------------
print '==> defining some tools'
-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector
collectgarbage()
a = cutorch.getDeviceProperties(1)
print('GPU free memory :'..tostring(a.freeGlobalMem))
parameters,gradParameters = model:getParameters()
a = cutorch.getDeviceProperties(1)
print('GPU free memory :'..tostring(a.freeGlobalMem))

----------------------------------------------------------------------
if opt.freezeLayers then
    print '==> freeze some layers parameters during training'
    modules = model.modules
    shouldFreezeLayers = torch.LongTensor(#modules):fill(0)
    for layerIdStr in string.gmatch(opt.freezeLayers, '([^,]+)') do
        layerId = tonumber(layerIdStr)
        shouldFreezeLayers[layerId] = 1
    end
    learningRates = torch.FloatTensor(gradParameters:size(1)):fill(1)

    x_index = 1
    for iLayer = 1,#modules do
        numParamsModule = 0
        if (modules[iLayer].weight) then
            numParamsModule = numParamsModule + modules[iLayer].weight:size(1)*modules[iLayer].weight:size(2)
        end
        if (modules[iLayer].bias) then
            numParamsModule = numParamsModule + modules[iLayer].bias:size(1)
        end
        if (shouldFreezeLayers[iLayer]==1) then
            print('freeze params of layer - ', iLayer)
            learningRates[{{x_index,(x_index + numParamsModule - 1)}}] = 0
        end
        print(iLayer, ' = ', x_index, ' : ', x_index + numParamsModule)
        x_index = x_index + numParamsModule
    end

    optimState.learningRates = learningRates
end
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
local totalErr = 0
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
        --a = cutorch.getDeviceProperties(1)
        collectgarbage()
        --print('GPU free memory :'..tostring(a.freeGlobalMem / a.totalGlobalMem))
        --print(inputs:size())
	    inputs = inputs:cuda()

	    -- create closure to evaluate f(X) and df/dX
	    local feval = function(x)
            -- get new parameters
            if x ~= parameters then
                print('DEBUG')
                print(parameters:size())
                parameters:copy(x)
            end

            -- reset gradients
            gradParameters:zero()

            -- evaluate function for complete mini batch
            -- estimate f
            local output = model:forward(inputs)
            local numInputs = inputs:size()[1]
            local err = criterion:forward(output, targets)

            -- f is the average of all criterions
            totalErr = totalErr + err

            -- estimate df/dW
            local df_do = criterion:backward(output, targets)
            model:backward(inputs, df_do)

            -- update confusion
            for i=1,numInputs do
              confusion:add(output[i], targets[i])
            end

            -- return f and df/dX
            return err,gradParameters
	    end

	    optimMethod(feval, parameters, optimState)
        -- grabage collection after every batch
        collectgarbage()
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

 print('Error = '..tostring(totalErr))
 -- update logger/plot
 trainLogger:add{['% total accuracy'] = confusion.totalValid * 100, 
     ['% average accuracy'] = confusion.averageValid * 100,
     ['cost function error'] = totalErr}
 if opt.plot then
     trainLogger:style{['% total accuracy'] = '-'}
     trainLogger:plot()
 end

 -- save/log current net
 print('==> saving model & state to '..state_file_path)
 os.rename(state_file_path, state_file_last_path)
 torch.save(state_file_path, model)

 -- next epoch
 confusion:zero()
 epoch = epoch + 1
end
