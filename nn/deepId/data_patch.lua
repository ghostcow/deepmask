package.path = package.path .. ";../?.lua"
require 'options'
require 'deep_id_utils'

----------------------------------------------------------------------
-- use -visualize to show network
-- parse command line arguments
if not opt then
    print '==> processing options'
    opt = getOptions()
end

local data_file_path = opt.dataPath
fileFormat = string.sub(data_file_path, -3)
local data_set
if (fileFormat == 'mat') then
    require 'mattorch'
    -- expect mat file
    data_set = mattorch.load(data_file_path)
elseif (fileFormat == '.t7') then
    -- expect torch file
    data_set = torch.load(data_file_path)
else
    error('unsupprted dataFormat option : '..string.sub(data_file_path, -3))
end

trsize = data_set.train:size()[4]
tesize = data_set.test:size()[4]

if opt.trainOnly then
    trainDataInner = {
        -- the original matlab format is nImages x 3 x height x width
        -- (where height=width=152)
        -- but it's loaded into torch like this : width x height x 3 x nImages

        data = torch.cat(data_set.train:transpose(1,4):transpose(2,3),
            data_set.test:transpose(1,4):transpose(2,3), 1),
        labels = torch.cat(data_set.trainLabels[1], data_set.testLabels[1]),
    }

    -- convert to our general dataset format
    trainData = {
        numChunks = 1,
        getChunk = function(iChunk) return trainDataInner end
    }
else
    trainDataInner = {
        -- the original matlab format is nImages x 3 x height x width
        -- (where height=width=152)
        -- but it's loaded into torch like this : width x height x 3 x nImages

        data = data_set.train:transpose(1,4):transpose(2,3),
        labels = data_set.trainLabels[1],
    }

    testDataInner = {
        data = data_set.test:transpose(1,4):transpose(2,3),
        labels = data_set.testLabels[1],
    }

    -- convert to our general dataset format
    trainData = {
        numChunks = 1,
        getChunk = function(iChunk) return trainDataInner end
    }
    testData = {
        numChunks = 1,
        getChunk = function(iChunk) return testDataInner end
    }
end

-- crop relevant patches
print('cropping patch number ', opt.patchIndex)
if opt.visualize then
    trainDataInnerOriginalData = trainDataInner.data:clone()
end

useFlippedPatches = true
if trainDataInner then
    trainDataInner.data = DeepIdUtils.getPatch(trainDataInner.data, opt.patchIndex, useFlippedPatches)
    if useFlippedPatches then
        -- replicate labels
        trainDataInner.labels = torch.cat(trainDataInner.labels, trainDataInner.labels)
    end
end

if testDataInner then
    -- TODO : should we add flipped patches to test set also ?
    testDataInner.data = DeepIdUtils.getPatch(testDataInner.data, opt.patchIndex, useFlippedPatches)
    if useFlippedPatches then
        -- replicate labels
        testDataInner.labels = torch.cat(testDataInner.labels, testDataInner.labels)
    end
end
trainDataInner.size = function() return trainDataInner.data:size(1) end
testDataInner.size = function() return testDataInner.data:size(1) end

-- classes - define classes array (used later for computing confusion matrix)
nLabels = trainDataInner.labels:max()
classes = {}
for i=1,nLabels do
    table.insert(classes, tostring(i))
end

------visualizing data---------------------------
if opt.visualize then
    nSamples = 100
    require 'gfx.js'
    print '==> visualizing data'
    if (require 'gnuplot') then
        gnuplot.figure(1)
        gnuplot.hist(trainDataInner.labels, trainDataInner.labels:max())
        gnuplot.title('#samples per label - training')
        gnuplot.figure(2)
        gnuplot.hist(testDataInner.labels, testDataInner.labels:max())
        gnuplot.title('#samples per label - test')
    end

    local firstSamplesTrain
    if useFlippedPatches then
        firstSamplesTrain = torch.Tensor(2*nSamples, trainDataInner.data:size(2),
            trainDataInner.data:size(3), trainDataInner.data:size(4))
        firstSamplesTrain[{{1, nSamples}}] = trainDataInner.data[{ {1,nSamples} }]
        firstSamplesTrain[{{nSamples+1, 2*nSamples}}] =
            trainDataInner.data[{ {trainDataInner:size()/2 + 1,trainDataInner:size()/2 + nSamples} }]
    else
        firstSamplesTrain = trainDataInner.data[{ {1,nSamples} }]
    end

    gfx.image(firstSamplesTrain, {legend='train - samples'})
    if not opt.trainOnly then
        local first100Samples_test = testDataInner.data[{ {1,nSamples} }]
        gfx.image(first100Samples_test, {legend='test - 100 samples'})
    end

    --- gfx is buggy, so we will save images into files...
    require 'image'
    for iImage = 1,firstSamplesTrain:size(1) do
        im = firstSamplesTrain[iImage]
        image.save('visualize/'..opt.patchIndex..'_'..iImage..'.jpg',im)
    end
    for iImage = 1,firstSamplesTrain:size(1) do
        fullIm = trainDataInnerOriginalData[iImage]
        image.save('visualize/'..'full_'..iImage..'.jpg',fullIm)
    end
end