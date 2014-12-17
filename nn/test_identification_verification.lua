package.path = package.path .. ';' .. paths.concat(paths.dirname(paths.thisfile()), 'train', '?.lua')


require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

----------------------------------------------------------------------
-- shuffle images while preserving balnced partition to match/mismatch pairs
shuffleTest = {}
for iChunk = 1,testData.numChunks do
    testDataChunk = testData.getChunk(iChunk)
    permFileName = paths.concat(paths.dirname(opt.dataPath), 'testPermutaion_'..iChunk..'.t7')
    if paths.filep(permFileName) then
        print '==> Loading test data permutation'
        shuffleTest[iChunk] = torch.load(permFileName)
    else
        print '==> Creating test data permutation'
        shuffle = RandomizePairs.randomizeImages(testDataChunk.labels)
        shuffleTest[iChunk] = shuffle
        torch.save(permFileName, shuffle)
    end
end

local nPairs = 0
local nPosPairs = 0
print '==> defining test procedure'
-- test function
function test()
    -- turn off dropout modules
    model:evaluate()

    -- local vars
    local time = sys.clock()

    -- test over test data
    print('==> testing on test set:')
    local totalSize = 0
    local testDataChunk
    local totalErrIden = 0
    local totalErrVer = 0
    for iChunk = 1,testData.numChunks do
        testDataChunk = testData.getChunk(iChunk)
        local chunkSize = testDataChunk:size()
        totalSize = totalSize + chunkSize
        shuffleChunk = shuffleTest[iChunk]
        for t = 1,testDataChunk:size(),opt.batchSize do
            -- disp progress
            xlua.progress(t, testDataChunk:size())
            --- create mini batch
            local inputs = torch.Tensor(opt.batchSize, 3, imageDim, imageDim)
            local targets = torch.Tensor(opt.batchSize)
            for i = t,(t+opt.batchSize-1) do
                -- apply mod operator, in order to complete last batch with first samples
                inputs[{i-t+1}] = testDataChunk.data[shuffleChunk[1 + (i-1)%chunkSize]]
                targets[{i-t+1}] = testDataChunk.labels[shuffleChunk[1 + (i-1)%chunkSize]]
            end
            collectgarbage()
            inputs = inputs:cuda()
            targets = targets:cuda()
            local numInputs = inputs:size(1)

            -- test samples
            features = model:forward(inputs)
            idenOutput = idenNet:forward(features)
            idenErr = idenCriterion:forward(idenOutput, targets)
            verErr = 0
            for iImage = 1,numInputs,2 do
                featuresPair = {features[{iImage}], features[{iImage+1}]}
                local labelPair
                nPairs = nPairs + 1
                if (targets[iImage] == targets[iImage+1]) then
                    labelPair = 1
                    nPosPairs = nPosPairs + 1
                else
                    labelPair = -1
                end
                verOutput = verNet:forward(featuresPair)
                verErr = verErr + verCriterion:forward(verOutput, labelPair)
            end
            totalErrIden = totalErrIden + idenErr
            totalErrVer = totalErrVer + verErr/(numInputs/2)

            local outputs = model:forward(inputs)
            for i=1,numInputs do
                confusion:add(idenOutput[i], targets[i])
            end

            -- grabage collection after every batch (TODO : might be too expensive...)
            collectgarbage()
        end
    end

    -- time taken
    time = sys.clock() - time
    time = time / totalSize
    print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

    -- print confusion matrix  & error funstion values
    print('number of positive pairs=',nPosPairs,'/',nPairs)
    confusion:updateValids()
    print("accuracy = ", confusion.totalValid * 100)
    local filename_confusion = paths.concat(opt.save, 'confusion_test')
    torch.save(filename_confusion, confusion)

    testLogger:add{['% total accuracy'] = confusion.totalValid * 100,
        ['% average accuracy'] = confusion.averageValid * 100,
        ['identification cost function error'] = totalErrIden,
        ['verification cost function error'] = totalErrVer,
        ['total cost function error'] = totalErrIden+lambda*totalErrVer}

    -- next epoch
    confusion:zero()

    -- turn on dropout modules
    model:training()
end
