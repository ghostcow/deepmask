package.path = package.path .. ";" .. paths.concat(paths.dirname(paths.thisfile()), 'train', '?.lua')
require 'torch'
require 'nn'
require 'xlua'
require 'optim'
require 'options'
-- some modules from train folder
require 'validate_data'
require 'balance_classes'
require 'freeze_layers'
require 'train_utils'
require 'cunn'
require 'cutorch'

useAssert = true
----------------------------------------------------------------------
-- parse command line arguments
if not opt then
    print '==> processing options'
    opt = getOptions()
end
----------------------------------------------------------------------
initializeTrainTools()
----------------------------------------------------------------------
print '==> defining cost function to NLL'
if opt.balanceClasses then
    criterion = nn.ClassNLLCriterion(getClassWeights())
else
    criterion = nn.ClassNLLCriterion()
end
criterion:cuda()

-- load pre-trained model if exist
if paths.filep(state_file_path_best) then
    print '==> loadind pre-trained network (best)'
    model = torch.load(state_file_path_best)
    print '==> model :'
    print(model)
elseif paths.filep(state_file_path) then
    print '==> loadind pre-trained network'
    model = torch.load(state_file_path)
    print '==> model :'
    print(model)
end

----------------------------------------------------------------------
print '==> defining some tools'

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
if not (opt.freezeLayers == '') then
    freezeLayers(opt.freezeLayers)
end
----------------------------------------------------------------------
print '==> defining training procedure'
function train()

model:training()

-- epoch tracker
epoch = epoch or 1

-- local vars
local time = sys.clock()

-- do one epoch
confusion:zero()
print('==> doing epoch on training data:')
print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
local totalErr = 0
local timeLoad
local trainDataChunk
for iChunk = 1,trainData.numChunks do
    timeLoad = sys.clock()
    trainDataChunk = trainData.getChunk(iChunk)

    timeLoad = sys.clock() - timeLoad
    print("\n==> chunk no. "..iChunk.." -data loading time = " .. timeLoad .. ' [s]')

    chunkSize = trainDataChunk:size()
    for iPass = 1,opt.numPassesPerChunk do
        print("\npass no. "..iPass.." for current chunk")
        -- shuffle at each epoch
        shuffle = torch.randperm(chunkSize)
        for t = 1,chunkSize,opt.batchSize do
            -- disp progress
            xlua.progress(t, chunkSize)

            -- create mini batch
            local inputs = torch.Tensor(opt.batchSize, 3, imageDim, imageDim)
            local targets = torch.Tensor(opt.batchSize)

            for i = t,(t+opt.batchSize-1) do
                -- apply mod operator, in order to complete last batch with first samples
                inputs[{i-t+1}] = trainDataChunk.data[shuffle[1 + (i-1)%chunkSize]]
                targets[{i-t+1}] = trainDataChunk.labels[shuffle[1 + (i-1)%chunkSize]]
            end
            -- NOTE: we suppport training on CUDA only
            --a = cutorch.getDeviceProperties(1)
            collectgarbage()
            --print('GPU free memory :'..tostring(a.freeGlobalMem / a.totalGlobalMem))
            -- print(inputs:size())
            inputs = inputs:cuda()
            targets = targets:cuda()

            -- create closure to evaluate f(X) and df/dX
            local feval = function(x)
                -- get new parameters
                if x ~= parameters then
                    print(parameters:size())
                    parameters:copy(x)
                end
                MyAssert(isValid(parameters), "non-valid model parameters", useAssert)

                -- reset gradients
                gradParameters:zero()

                -- evaluate function for complete mini batch
                -- estimate f
                MyAssert(isValid(inputs), "there are some nan's in inputs", useAssert)
                local output = model:forward(inputs)
                if (not isValid(output)) then
                    -- invalid output
                    local midInput = inputs:clone()
                    local midOutput
                    for iLayer = 1,#(model.modules) do
                        midOutput = model:get(iLayer):forward(midInput)
                        if not isValid(midOutput) then
                            --print(iLayer, midOutput:size())
                            --print(midOutput[{{},1}])
                            MyAssert(false, "there are some nan's in output", useAssert)
                        end
                        midInput = midOutput
                    end
                end

                local numInputs = inputs:size()[1]
                local err = criterion:forward(output, targets)

                -- f is the average of all criterions
                totalErr = totalErr + err

                -- estimate df/dW
                local df_do = criterion:backward(output, targets)
                if (not isValid(df_do)) then
                    print 'after criterion:backward - non-valid model df_do'
                    -- df_do:zero()
                end

                model:backward(inputs, df_do)

                if (not isValid(gradParameters)) then
                    print(err)
                    print 'after model:backward - non-valid model gradParameters'
                    -- gradParameters:zero()
                end

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
end

 -- time taken
 time = sys.clock() - time
 print("\n==> epoch learn time = " .. time/opt.numPassesPerChunk .. ' [s]')

-- print confusion matrix  & error funstion values
 confusion:updateValids()
 print("accuracy = ", confusion.totalValid * 100)
 local filename_confusion = paths.concat(opt.save, 'confusion_train')
 torch.save(filename_confusion, confusion)
 print('Error = '..(totalErr/opt.numPassesPerChunk))
 trainLogger:add{['% total accuracy'] = confusion.totalValid * 100, 
     ['% average accuracy'] = confusion.averageValid * 100,
     ['cost function error'] = totalErr/opt.numPassesPerChunk}
 if opt.plot then
     trainLogger:style{['% total accuracy'] = '-'}
     trainLogger:plot()
 end

 -- check all model parameters validity
 MyAssert(isValid(parameters), "non-valid model parameters", useAssert)
 MyAssert(isValid(gradParameters), "non-valid model gradParameters", useAssert)

 -- save/log current net
 print('==> saving model & state to '..state_file_path)
 os.rename(state_file_path, state_file_last_path)
 torch.save(state_file_path, model)

 -- next epoch
 confusion:zero()
 epoch = epoch + 1
end
