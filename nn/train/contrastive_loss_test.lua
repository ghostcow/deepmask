require 'nn'
require 'optim'
require 'paths'

-- small wrapper to optim.sgd, with more convenient syntax
function doSgd(x, dfdx, optimState)
    local opfunc = function()
        return 0,dfdx
    end
    local fx_,dfdx_ = opfunc(x)
    optim.sgd(opfunc, x, optimState)
end

--- constants
batchSize = 64
inputSize = 5
featureDim = 160
nLabels = 10

pairLabel = 1
personLabel = 1

optimStateIden = {learningRate = 0.001, momentum = 0.5 }
optimState = {learningRate = 0.001, momentum = 0.5}

--- cost criterions
lambda = 0.05
marginValue = 5
verCriterion = nn.HingeEmbeddingCriterion(marginValue)
idenCriterion = nn.ClassNLLCriterion()

--- core network generating face features
model = nn.Linear(inputSize, featureDim)

--- top level "network" for verification
verNet = nn.PairwiseDistance(2)

--- top level "network" for identification
idenNet = nn.Sequential()
idenNet:add(nn.Linear(featureDim, nLabels))
idenNet:add(nn.LogSoftMax())

-- Use a typical generic gradient update function
netParams,netGradParams = model:getParameters()
idenNetParams,idenNetGradParams = idenNet:getParameters()
function gradUpdate(inputs, targets)
    features = model:forward(inputs)

    idenNet:zeroGradParameters()
    verNet:zeroGradParameters()
    model:zeroGradParameters()
    -- softmax loss (identification)
    idenOutput = idenNet:forward(features)
    idenErr = idenCriterion:forward(idenOutput, targets)
    idenCritGrad = idenCriterion:backward(idenOutput, targets)
    idenNetGradInput = idenNet:backward(features, idenCritGrad)
    -- gradient-descent for identifiaction network
    doSgd(idenNetParams, idenNetGradParams, optimStateIden)

    -- contrastive loss (verification)
        -- run over all pairs {i,i+1}
    verNetGradInput = torch.Tensor():resizeAs(idenNetGradInput):fill(0)
    local numInputs = inputs:size()[1]
    for iImage = 1,(numInputs-1),2 do
        featuresPair = {features[{iImage}], features[{iImage+1}] }
        verOutput = verNet:forward(featuresPair)
        verErr = verCriterion:forward(verOutput, pairLabel)
        verCritGrad = verCriterion:backward(verOutput, pairLabel)
        verNetPairGradParams = verNet:backward(featuresPair, verCritGrad)

        verNetGradInput[{iImage}] = verNetPairGradParams[1]
        verNetGradInput[{iImage+1}] = verNetPairGradParams[2]
    end

    -- backward combined loss over the core network
    combinedGrad = idenNetGradInput + verNetGradInput:mul(lambda)
    model:backward(inputs, combinedGrad)
    doSgd(netParams, netGradParams, optimState)
end

inputs = torch.rand(batchSize, inputSize)
targets = torch.Tensor(batchSize):fill(personLabel)
gradUpdate(inputs, targets)