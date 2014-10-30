--- relevant options : save, dataPath

require 'options'
require 'nn'
require 'cunn'
require 'ccn2'
require 'mattorch'
require 'optim'
require 'xlua'

---Load model -----------------------------------------------------------------------------------------------
opt = getOptions()
opt.save = paths.concat('../results/', opt.save)
local state_file_path = paths.concat(opt.save, 'model.net')
model = torch.load(state_file_path)
for iModule = 1,#model.modules do
    if (torch.type(model.modules[iModule]) == 'nn.Dropout') then
        model.modules[iModule].train = false    
    end
end

dofile 'data.lua'
confusion = optim.ConfusionMatrix(classes)

-- test function
function testTop5()
    numInputTotal = 0
    numTrue = 0
    numTrueTop5 = 0
    confusion:zero()

    -- local vars
    local time = sys.clock()

    -- test over test data
    print('==> testing on test set:')
    local totalSize = 0
    local testDataChunk
    for iChunk = 1,testData.numChunks do
        testDataChunk = testData.getChunk(iChunk)
        totalSize = totalSize + testDataChunk:size()
        for t = 1,testDataChunk:size(),opt.batchSize do
            xlua.progress(t, testDataChunk:size())
            if ((t+opt.batchSize-1) > testDataChunk:size()) then
                -- we don't use the last samples
                break
            end

            -- get new sample
            local inputs = testDataChunk.data[{{t,t+opt.batchSize-1}}]
            local numInputs = inputs:size()[1]
            inputs = inputs:cuda()
            local targets = testDataChunk.labels[{{t,t+opt.batchSize-1}}]

            -- test sample
            local outputs = model:forward(inputs)

            numInputTotal = numInputTotal + numInputs
            for i=1,numInputs do
                local realLabel = targets[i]
                local classScores = outputs[i]
                classScores = classScores:float()

                confusion:add(classScores, realLabel)

                _,indices = torch.sort(classScores, 1, true)
                if (indices[1] == realLabel) then
                    numTrue = numTrue + 1
                end

                for iClass = 1,5 do
                    if (indices[iClass] == realLabel) then
                        numTrueTop5 = numTrueTop5 + 1
                        break
                    end
                end
            end

            -- grabage collection after every batch (TODO : might be too expensive...)
            collectgarbage()
        end
    end

    -- timing
    time = sys.clock() - time
    time = time / totalSize
    print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

    confusion:updateValids()
    print("\n==> total accuracy = " .. numTrue/numInputTotal)
    print("\n==> total accuracy (confusion matrix) = " .. confusion.totalValid)
    print("\n==> top5 accuracy = " .. numTrueTop5/numInputTotal)
end

testTop5()
