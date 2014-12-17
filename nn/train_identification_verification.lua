package.path = package.path .. ';' .. paths.concat(paths.dirname(paths.thisfile()), 'train', '?.lua')
require 'torch'
require 'nn'
require 'xlua'
require 'optim'
require 'options'
require 'math'
require 'train_utils'
require 'randomize_pairs'

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
    print '==> processing options'
    opt = getOptions()
end
----------------------------------------------------------------------
initializeTrainTools()
-- create different optimState for the identification network
optimStateIden = {}
for k,v in pairs(optimState) do
    optimStateIden[k] = v
end
----------------------------------------------------------------------
if opt.loadState then
    loadedModels = torch.load(state_file_path)
    model = loadedModels.model
    idenNet = loadedModels.idenNet
    margin = loadedModels.margin
else
    margin = opt.margin  -- intial margin value for the verification cost function

    -- check if the model contains a classification layer at the end
    -- if it does, cut it out into seperate identification layer
    nModules = #model.modules
    if (torch.type(model.modules[nModules]) == 'nn.Linear') then
        if (model.modules[nModules].weight:size(1) == nLabels) then
            -- this is a classification layer
            local model_representaion = nn.Sequential()
            for iModule = 1,(nModules-1) do
                model_representaion:add(model.modules[iModule])
            end
            model = model_representaion
            collectgarbage()
        end
    end
    model:cuda()

    -- top level "network" for identification
    idenNet = nn.Sequential()
    idenNet:add(nn.Linear(featureDim, nLabels))
    idenNet:add(nn.LogSoftMax())
    idenNet:cuda()
end
lambda = opt.lambda -- weight factor of the verificatoin cost
distNorm = 2  -- using L2 norm for the verification loss

-- top level "network" for verification
verNet = nn.PairwiseDistance(2)
verNet:cuda()
verCriterion = nn.HingeEmbeddingCriterion(margin):cuda()
idenCriterion = nn.ClassNLLCriterion():cuda()

--- training procedure
netParams,netGradParams = model:getParameters()
idenNetParams,idenNetGradParams = idenNet:getParameters()

nImages = opt.batchSize*math.ceil(trainData.size()/opt.batchSize)
nPairs = nImages/2
pairsDists = torch.Tensor(nPairs):fill(0)
pairsLabels = torch.Tensor(nPairs):fill(0)

print '==> defining training procedure'
function train()
    -- epoch tracker
    epoch = epoch or 1

    -- local vars
    local time = sys.clock()

    -- do one epoch
    print('==> doing epoch on training data:')
    print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
    totalSize = 0
    totalErrIden = 0 -- identification loss value after full epoch
    totalErrVer = 0  -- verification loss value after full epoch
    nPosPairs = 0
    for iChunk = 1,trainData.numChunks do
        trainDataChunk = trainData.getChunk(iChunk)
        chunkSize = trainDataChunk:size()
        totalSize = totalSize + chunkSize

        -- shuffle images while preserving balnced partition to match/mismatch pairs
        shuffle = RandomizePairs.randomizeImages(trainDataChunk.labels)

        for t = 1,chunkSize,opt.batchSize do
            batchStartIndex = (iChunk-1)*chunkSize + t

            -- disp progress
            xlua.progress(t, chunkSize)

            --- create mini batch
            inputs = torch.Tensor(opt.batchSize, 3, imageDim, imageDim)
            targets = torch.Tensor(opt.batchSize)
            for i = t,(t+opt.batchSize-1) do
                -- apply mod operator, in order to complete last batch with first samples
                inputs[{i-t+1}] = trainDataChunk.data[shuffle[1 + (i-1)%chunkSize]]
                targets[{i-t+1}] = trainDataChunk.labels[shuffle[1 + (i-1)%chunkSize]]
            end
            collectgarbage()
            inputs = inputs:cuda()
            targets = targets:cuda()

            --- forward & backward passes
            features = model:forward(inputs)

            -- softmax loss (identification)
            idenOutput = idenNet:forward(features)
            idenErr = idenCriterion:forward(idenOutput, targets)
            idenCritGrad = idenCriterion:backward(idenOutput, targets)
            idenNet:zeroGradParameters()
            idenNetGradInput = idenNet:backward(features, idenCritGrad)
            doSgd(idenNetParams, idenNetGradParams, optimStateIden) -- gradient-descent for identifiaction network

            -- contrastive loss (verification)
                -- run over all pairs {i,i+1}
            verNetGradInput = torch.CudaTensor():resizeAs(idenNetGradInput)
            local numInputs = inputs:size(1)
            verErr = 0
            for iImage = 1,numInputs,2 do
                featuresPair = {features[{iImage}], features[{iImage+1}]}
                local labelPair
                if (targets[iImage] == targets[iImage+1]) then
                    labelPair = 1
                    nPosPairs = nPosPairs + 1
                else
                    labelPair = -1
                end

                verOutput = verNet:forward(featuresPair)
                verErr = verErr + verCriterion:forward(verOutput, labelPair)
                verCritGrad = verCriterion:backward(verOutput, labelPair)
                verNetPairGradParams = verNet:backward(featuresPair, verCritGrad)

                verNetGradInput[{iImage}] = verNetPairGradParams[1]
                verNetGradInput[{iImage+1}] = verNetPairGradParams[2]

                -- gathering all pairs dists
                iPair = (batchStartIndex + iImage) / 2
                pairsDists[iPair] = featuresPair[1]:dist(featuresPair[2], distNorm)
                pairsLabels[iPair] = labelPair
            end

            -- backward combined loss over the core network
            combinedGrad = idenNetGradInput:add(verNetGradInput:mul(lambda))
            model:zeroGradParameters()
            model:backward(features, combinedGrad)
            doSgd(netParams, netGradParams, optimState)

            -- update confusion matrix & error costs
            for i=1,numInputs do
                confusion:add(idenOutput[i], targets[i])
            end
            totalErrIden = totalErrIden + idenErr
            totalErrVer = totalErrVer + verErr/(numInputs/2)
        end
    end

    -- update margin value - take the best threshold based on pairsDists & pairsLabels
    distsSorted,sortedDistsIndices = torch.sort(pairsDists)
    pairsLabelsSorted = pairsLabels:index(1,sortedDistsIndices)
    -- based on distsSorted, we will check all thresholds and choose the best one
    isPosClass = torch.Tensor(nPairs):copy(pairsLabels):apply(function(x) if (x==1) then return 1 else return 0 end end)
    isNegClass = torch.Tensor(nPairs):copy(pairsLabels):apply(function(x) if (x==-1) then return 1 else return 0 end end)

    -- error :
    -- FP - negative pair with small distnace (below th)
    -- FN - positive pair with high distnace (above th)
    errors = torch.cumsum(isNegClass) + nPosPairs - torch.cumsum(isPosClass)
    minError,thIndex = torch.min(errors, 1)
    thIndex = thIndex[1]
    if (thIndex == 1) then
        -- rare case no. 1
        verCriterion.margin = distsSorted[1]/2
    elseif (thIndex == nPairs) then
        -- rare case no. 2
        verCriterion.margin = distsSorted[nPairs] + 1
    else
        verCriterion.margin = distsSorted[thIndex]
    end

    -- time taken
    time = sys.clock() - time
    time = time / totalSize
    print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

    -- print confusion matrix  & error funstion values
    print('number of positive pairs=',nPosPairs,'/',nPairs)
    print('new margin value=',verCriterion.margin, ',minimizing error to:',errors[thIndex]/nPairs)
    confusion:updateValids()
    print("accuracy = ", confusion.totalValid * 100)
    local filename_confusion = paths.concat(opt.save, 'confusion_train')
    torch.save(filename_confusion, confusion)

    trainLogger:add{['% total accuracy'] = confusion.totalValid * 100,
        ['% average accuracy'] = confusion.averageValid * 100,
        ['identification cost function error'] = totalErrIden,
        ['verification cost function error'] = totalErrVer,
        ['total cost function error'] = totalErrIden+lambda*totalErrVer}

    -- save/log current net
    print('==> saving model & state to '..state_file_path)
    os.rename(state_file_path, state_file_last_path)
    torch.save(state_file_path, {model=model,idenNet=idenNet, margin=margin})

    -- next epoch
    confusion:zero()
    epoch = epoch + 1
end
