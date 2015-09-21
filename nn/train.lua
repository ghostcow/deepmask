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
require 'blur'

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
    print '==> processing options'
    opt = getOptions()
end

----------------------------------------------------------------------
print '==> defining some tools'
-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(dataset.classes)

-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector
parameters,gradParameters = net:getParameters()

----------------------------------------------------------------------
print '==> configuring optimizer - SGD'

optimState = {}

if opt.loadState ~= 'none' then
    optimState = torch.load(opt.loadState)
end

optimState.learningRate = opt.learningRate
optimState.momentum = opt.momentum
optimState.weightDecays = getWeightDecays(opt.weightDecay, parameters, gradParameters)

----------------------------------------------------------------------
print '==> defining training procedure'

t = 0 -- batch counter
totalErr = 0 -- totalErr accumelator
function trainBatch(inputs, targets)
    inputs = inputs:cuda()
    targets = targets:double():cuda()

    -- disp progress
    xlua.progress(t, dataset:sizeTrain())
    t = t + inputs:size(1)

    -- create closure to evaluate f(X) and df/dX
    local feval = function(x)
        -- get new parameters
        if x ~= parameters then
            print(parameters:size())
            parameters:copy(x)
        end

        -- reset gradients
        net:zeroGradParameters()
        gradParameters:zero()

        -- evaluate function for complete mini batch - estimate f
        local output = net:forward(inputs)
        local err = criterion:forward(output, targets)

        -- estimate df/dW
        local df_do = criterion:backward(output, targets)
        net:backward(inputs, df_do)

        -- update confusion
        confusion:batchAdd(output, targets)

        -- update total err
        totalErr = totalErr + err

        -- return f and df/dX
        return err,gradParameters
    end

    optim.sgd(feval, parameters, optimState)
    if net.syncParameters ~= nil then
        net:syncParameters()
    end

    -- grabage collection after every batch
    collectgarbage()
end

function train()
    net:training()

    -- epoch tracker
    epoch = epoch or 1

    -- local vars
    local timer = torch.Timer()
    totalErr = 0
    t = 0

    -- do one epoch
    print('==> doing epoch on training data:')
    print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

    -- each epoch we shuffle the data
    local perm = torch.randperm(dataset.trainIndicesSize):long()
    local epoch_marker = epoch
    dataset:shuffle(perm)

    for i=1,dataset:sizeTrain(),opt.batchSize do
        donkeys:addjob(
            -- the job callback
            function()
                -- shuffle the data in each thread, every epoch
                if tepoch ~= epoch_marker then
                    tepoch = epoch_marker
                    dataset:shuffle(perm)
                end

                local maxIndex = math.min(dataset:sizeTrain(), i + opt.batchSize - 1)
                local inputs, targets, _ = dataset:get(i, maxIndex, dataset.trainIndices)

                local sigma = opt.blurSigma - opt.blurSigma/opt.epochs*(tepoch-1)
                if opt.blurSize ~= -1 and sigma > 0 then
                    inputs = nn.Blur(opt.blurSize, sigma):forward(inputs)
                end

                collectgarbage()
                return inputs, targets
            end,

            -- the end callback
            -- ran in the main thread
            trainBatch)
    end

    -- wait for all jobs to finish
    print("Waiting for jobs to finish!")
    donkeys:synchronize()

    -- time taken
    local time = timer:time().real / dataset:sizeTrain()
    print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

    -- print confusion matrix  & error funstion values
    logConfusion(confusion, totalErr)
    -- save current network
    logNetwork(model, optimState)

    -- check all model parameters validity
    MyAssert(isValid(parameters), "non-valid model parameters")
    MyAssert(isValid(gradParameters), "non-valid model gradParameters")

    -- next epoch
    confusion:zero()
    epoch = epoch + 1
end
