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
local tx = require'pl.tablex'
local tds = require 'tds'
tds.hash.__ipairs = tds.hash.__pairs

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
maskParameters, maskGradParameters = mask:getParameters()
scoreParameters, scoreGradParameters = score:getParameters()

----------------------------------------------------------------------
print '==> configuring optimizer - SGD'

optimState = {}

if opt.loadState ~= 'none' then
    optimState = torch.load(opt.loadState)
end

optimState.learningRate = opt.learningRate
optimState.momentum = opt.momentum
optimState.weightDecay = opt.weightDecay

optimStateMask = tx.copy(optimState)
optimStateScore = tx.copy(optimState)

----------------------------------------------------------------------
print '==> defining training procedure'

t = 0 -- batch counter
totalErr = 0 -- totalErr accumelator
function trainBatch(branch, classes, inputs, masks)
    inputs = inputs:cuda()

    -- current progress
    xlua.progress(t, sizeTrain())

    -- create closure to evaluate f(X) and df/dX
    local feval = function(x)
        -- TODO: eliminate code duplication
        -- get new parameters
        if branch == 1 then
            if x ~= maskParameters then
                maskParameters:copy(x)
            end
        else
            if x ~= scoreParameters then
                scoreParameters:copy(x)
            end
        end

        -- evaluate function for complete mini batch - estimate f
        local err
        if branch == 1 then
            mask:zeroGradParameters()

            masks = masks:cuda()
            local outputs = mask:forward(inputs)
            err = maskCriterion:forward(outputs, masks)
            local df_do = maskCriterion:backward(outputs, masks)
            mask:backward(inputs, df_do)

            return err, maskGradParameters
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
            return err, scoreGradParameters
        end
    end

    if branch == 1 then
        optim.sgd(feval, maskParameters, optimStateMask)
    else
        optim.sgd(feval, scoreParameters, optimStateScore)
    end

    -- update progress
    t = t + inputs:size(1)
    xlua.progress(t, sizeTrain())

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
    -- save current networks
--    logNetwork(mask, 'deepmask_mask')
--    logNetwork(score, 'deepmask_score')
    -- save optim states
--    logOptimState(optimStateMask, 'deepmask_mask')
--    logOptimState(optimStateScore, 'deepmask_score')

    -- check all model parameters validity
    -- just print warning for now
--    useAssert=false
--    MyAssert(isValid(maskParameters), "non-valid model parameters")
--    MyAssert(isValid(maskGradParameters), "non-valid model gradParameters")
--    MyAssert(isValid(scoreParameters), "non-valid model parameters")
--    MyAssert(isValid(scoreGradParameters), "non-valid model gradParameters")

    -- next epoch
    confusion:zero()
    epoch = epoch + 1
end
