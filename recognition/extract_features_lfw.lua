----------------------------------------------------------------------
-- Run the network over the given dataset
----------------------------------------------------------------------

package.path = package.path .. ";" .. '../nn_utils/?.lua'
require 'dataset'
require 'mattorch'
require 'paths'
require 'xlua'
require 'nn'
require 'cunn'
require 'cudnn'
require 'stn'
ffi = require 'ffi'

torch.setnumthreads(12)
torch.manualSeed(11)
torch.setdefaulttensortype('torch.FloatTensor')

function getOptions()
    local cmd = torch.CmdLine()

    cmd:text()
    cmd:text('Face feature extractor')
    cmd:text()
    cmd:text('Options:')

    cmd:option('--lfwPath', 'lfw_dataset.t7', 'path to lfw directory or dataset')
    cmd:option('--lfwProtocolPath', 'lfw_protocols', 'lfw pairs.txt file path')
    cmd:option('--modelPath', '../results/nn/model_best.net', 'path to nn model')
    cmd:option('--outputPath', '../results/features', 'path to lfw directory or dataset')
    cmd:option('--cpu', false, 'should use cpu')
    cmd:option('--batchSize', 128, 'batch size')

    local opt = cmd:parse(arg or {})
    return opt
end

function getDatapath(lfwPath)
    print '==> loading lfw dataset'
    local lfw_dataset
    if paths.filep(lfwPath) then
        lfw_dataset = torch.dataset.load(nil, lfwPath)
    else
        lfw_dataset = torch.dataset{paths={lfwPath}, sampleSize={1,100,100}, split=0 }
        lfw_dataset:save('lfw_dataset.t7')
    end

    return lfw_dataset
end

function loadModel(modelPath)
    print '==> loading model'
    local model = torch.load(modelPath)
    if not opts.cpu then
        model:cuda()
    end
    model:evaluate()

    -- remove classifier layers
    local n = 1
    while (torch.type(model.modules[n]) ~= 'nn.Linear') and (n <=  #model.modules) do
        n = n + 1
    end
    -- for full_conv model start from i=31
    for i = n,#model.modules do
        model.modules[i] = nil
    end

    return model
end

function getEmbeddingSize(model)
    local layer, _ = model:findModules('nn.Reshape')

    if #layer == 0 then
        layer, _ = model:findModules('nn.View')
        return layer[1].numElements
    end

    return layer[1].nelement
end

function extract_features(model, imagesData)
    local features = torch.FloatTensor(imagesData:size(1), getEmbeddingSize(model))

    local batchSize
    if opts.cpu then
        batchSize = 1
    else
        batchSize = 128
    end

    for i = 1,imagesData:size(1), batchSize do
        xlua.progress(i, imagesData:size(1))
        local lastIndex = math.min(imagesData:size(1), i + batchSize - 1)
        local inputs = imagesData[{{i, lastIndex}}]:clone():float() -- change to CUDA

        if not opts.cpu then
            inputs = inputs:cuda()
        end

        features[{{ i , lastIndex }}]:copy(model:forward(inputs):float())
        collectgarbage()
    end

    return features
end

function get_image_from_dataset(dataset, className, imageId)
    local classId = dataset.classIndices[className]
    local imageIdStr = string.format("%04d", imageId)
    local classImageList = dataset.classListTest[classId]

    if classImageList == nil then
        return torch.FloatTensor(1,100,100):zero(), false   
    end

    for i=1,classImageList:nElement() do
        local imagePath = ffi.string(torch.data(dataset.imagePath[classImageList[i]]))

        if string.match(imagePath, imageIdStr) then
            return dataset:defaultSampleHook(imagePath), true
        end
    end

    return torch.FloatTensor(1,100,100):zero(), false
end

function parse_lfw_pairs(dataset, lfwPairsPath, outputPath, model)
    local pairFile = io.open(lfwPairsPath, "r")

    local lines = pairFile:lines()
    local folds, foldSize = lines():match("(%d+)\t(%d+)")

    -- create pairs label according to pair.txt protocol
    local targetY = torch.CharTensor(folds*foldSize*2);
    for i=1,folds do
        local foldStartIndex = (i-1)*foldSize*2

        targetY[{{foldStartIndex + 1, foldStartIndex + foldSize}}]:fill(1)
        targetY[{{foldStartIndex + foldSize + 1, foldStartIndex + 2*foldSize}}]:fill(-1)
    end

    -- create pairs fold id according to pair.txt protocol
    local targetFoldId = torch.CharTensor(folds*foldSize*2);
    for i=1,folds do
        local foldStartIndex = (i-1)*foldSize*2

        targetFoldId[{{foldStartIndex + 1, foldStartIndex + foldSize*2}}]:fill(i)
    end

    -- get extracted features for each pair according to pair.txt protocol
    local images1 = torch.DoubleTensor(folds*foldSize*2, 1, 100, 100);
    local images2 = torch.DoubleTensor(folds*foldSize*2, 1, 100, 100);
    local check1, check2;
    for i=1,folds do
        local foldStartIndex = (i-1)*foldSize*2

        for samePairIndex=1,foldSize do
            local name, firstImage, secondImage = lines():match("(.*)\t(%d+)\t(%d+)")
            images1[{foldStartIndex + samePairIndex}], check1 = get_image_from_dataset(dataset, name, firstImage)
            images2[{foldStartIndex + samePairIndex}], check2 = get_image_from_dataset(dataset, name, secondImage)
            if check1==false or check2==false then
                targetY[{foldStartIndex + samePairIndex}] = 0;
            end
        end

        for diffPairIndex=1,foldSize do
            local name1, firstImage, name2, secondImage = lines():match("(.*)\t(%d+)\t(.*)\t(%d+)")
            images1[{foldStartIndex + foldSize + diffPairIndex}], check1 = get_image_from_dataset(dataset, name1, firstImage)
            images2[{foldStartIndex + foldSize + diffPairIndex}], check2 = get_image_from_dataset(dataset, name2, secondImage)
            if check1==false or check2==false then
                targetY[{foldStartIndex + foldSize + diffPairIndex}] = 0;
            end
        end
    end

    mattorch.save(outputPath, {targetX1 = extract_features(model, images1):double(),
                               targetX2 = extract_features(model, images2):double(),
                               targetY = targetY:double(),
                               foldId = targetFoldId:double()})
end

function parse_lfw_people(lfwPeoplePath)
    local peopleFile = io.open(lfwPeoplePath, "r")

    local lines = peopleFile:lines()
    local groupsNum = lines():match("(%d+)")
    local groups = {}

    for i=1,groupsNum do
        local groupSize = lines():match("(%d+)")
        groups[i] = {}

        for j=1,groupSize do
            local name, amount = lines():match("(.*)\t(%d+)")
            groups[i][name] = amount
        end
    end

    return groups
end

function extract_lfw_people_features(dataset, peopleProtocol, outputPath, model)
    -- count number of total images, 10 is the number of groups
    local acc = 0
    for foldId=1,10 do
        for _,v in next,peopleProtocol[foldId],nil do
            acc = acc + v
        end
    end

    -- create pairs label according to pair.txt protocol
    local targetY = torch.DoubleTensor(acc);
    local targetX = torch.DoubleTensor(acc, 1, 100, 100);
    local targetFoldId = torch.DoubleTensor(acc);

    local imageCounter = 1
    -- iterate over all folds
    for foldId=1,10 do
        -- iterate over all people inside a fold
        for className,imagesToRead in next,peopleProtocol[foldId],nil do
            -- iterate over all images of an identity
            for i=1,imagesToRead do
                local classId = dataset.classIndices[className]
                if not classId then
                    classId = 0
                end
                targetY[{imageCounter}] = classId
                targetX[{imageCounter}] = get_image_from_dataset(dataset, className, i)
                targetFoldId[{imageCounter}] = foldId
                imageCounter = imageCounter + 1
            end
        end
    end

    mattorch.save(outputPath, {targetX = extract_features(model, targetX):double(),
                               targetY = targetY,
                               foldId = targetFoldId})
end

function main()
    opts = getOptions()
    local lfw_dataset = getDatapath(opts.lfwPath)
    local model = loadModel(opts.modelPath)

    print('==> creating feature directory: ' .. opts.outputPath)
    paths.mkdir(opts.outputPath)

    print '==> extracting features for LFW pairs'
    parse_lfw_pairs(lfw_dataset, opts.lfwProtocolPath .. '/pairs.txt', opts.outputPath .. '/lfw_pair_feature.mat', model)

    print '==> extracting features for LFW people'
    local peopleProtocol = parse_lfw_people(opts.lfwProtocolPath .. '/people.txt')
    extract_lfw_people_features(lfw_dataset, peopleProtocol, opts.outputPath .. '/lfw_people_feature.mat', model)
end


main()
