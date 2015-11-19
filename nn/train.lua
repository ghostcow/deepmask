----------------------------------------------------------------------
-- Deepface nn train code torch7
--
-- Uses CUDA
----------------------------------------------------------------------
require 'torch'
require 'cutorch'
require 'xlua'
require 'optim'
require 'options'
require 'logger'
require 'assert'
require 'weight_decays'
require 'utils'

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
    print '==> processing options'
    opt = getOptions()
end

----------------------------------------------------------------------
print '==> defining some tools'
-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(getClasses())

-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector
maskParameters,maskGradParameters = mask:getParameters()
scoreParameters,scoreGradParameters = score:getParameters()

----------------------------------------------------------------------
print '==> configuring optimizer - SGD'

optimState = {}

if opt.loadState ~= 'none' then
    optimState = torch.load(opt.loadState)
end

optimState.learningRate = opt.learningRate
optimState.momentum = opt.momentum
optimState.weightDecay = opt.weightDecay

----------------------------------------------------------------------
print '==> defining training procedure'

t = 0 -- batch counter
totalErr = 0 -- totalErr accumelator
function trainBatch(inputs, masks, classes, branch)
    inputs = inputs:cuda()

    -- disp progress
    xlua.progress(t, sizeTrain())
    t = t + inputs:size(1)

    -- create closure to evaluate f(X) and df/dX
    local feval = function(x)
        -- get new parameters
        if x ~= parameters then
            print(parameters:size())
            parameters:copy(x)
        end

        -- evaluate function for complete mini batch - estimate f
        local err
        if branch == 1 then
            mask:zeroGradParameters()

            masks = masks:cuda()
            local outputs = mask:forward(inputs)
            err = maskCriterion:forward(outputs, masks) -- TODO: normalize the loss somehow
            local df_do = maskCriterion:backward(output, masks)
            mask:backward(inputs, df_do)

            return err,maskGradParameters
        else
            score:zeroGradParameters()

            classes = classes:cuda()
            local outputs = score:forward(inputs)
            err = scoreCriterion:forward(outputs, classes)
            local df_do = scoreCriterion:backward(outputs, classes)
            score:backward(inputs, df_do)

            -- update confusion
            confusion:batchAdd(outputs, classes)

            -- return f and df/dX
            return err,scoreGradParameters
        end
    end

    if branch == 1 then
        optim.sgd(feval, maskParameters, optimState)
    else
        optim.sgd(feval, scoreParameters, optimState)
    end

    -- garbage collection after every batch
    collectgarbage()
end

function train()
    mask:training()
    score:training()

    -- epoch tracker
    epoch = epoch or 1

    -- local vars
    local timer = torch.Timer()
    t = 0

    -- do one epoch
    print('==> doing epoch on training data:')
    print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

    for _=1,opt.epochSize do
        workers:addjob(
            -- the job callback
            function()
                if torch.uniform() > 0.5 then
                    return dataset:get(opt.batchSize,1)
                else
                    return dataset:get(opt.batchSize,2)
                end
            end,

            -- the end callback
            -- ran in the main thread
            trainBatch)
    end

    -- wait for all jobs to finish
    print("Waiting for jobs to finish!")
    workers:synchronize()

    -- time taken
    local time = timer:time().real / sizeTrain()
    print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

    -- print confusion matrix  & error funstion values
    logConfusion(confusion, totalErr)
    -- save current network
    logNetwork(net, optimState)

    -- check all model parameters validity
    MyAssert(isValid(parameters), "non-valid model parameters")
    MyAssert(isValid(gradParameters), "non-valid model gradParameters")

    -- next epoch
    confusion:zero()
    epoch = epoch + 1
end
