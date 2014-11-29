package.path = package.path .. ";../?.lua"
require 'options'
require 'deep_id_utils'

----------------------------------------------------------------------
-- use -visualize to show network
-- parse command line arguments
useFlippedPatches = true

if not opt then
    print '==> processing options'
    opt = getOptions()
end

local dataFilePath = opt.dataPath
local dataFileName
fileFormat = string.sub(dataFilePath, -3)
local loadFunc
local data_set
if (fileFormat == 'mat') then
    require 'mattorch'
    -- expect mat file
    loadFunc = mattorch.load
    dataFileName = dataFilePath:sub(1,-5)
elseif (fileFormat == '.t7') then
    fileFormat = 't7'
    -- expect torch file
    loadFunc = torch.load
    dataFileName = dataFilePath:sub(1,-4)
else
    error('unsupprted dataFormat option : '..string.sub(dataFilePath, -3))
end

trainDataInner = {
    data = torch.Tensor(),
    labels = torch.Tensor()
}
testDataInner = {
    data = torch.Tensor(),
    labels = torch.Tensor()
}

if paths.filep(dataFilePath) then
    -- all data in one file
    print('Loading from: ', dataFilePath)
    data_set = loadFunc(dataFilePath)
    -- the original matlab format is nImages x 3 x height x width
    -- (where height=width=152)
    -- but it's loaded into torch like this : width x height x 3 x nImages
    trainDataInner.data = data_set.train:transpose(1,4):transpose(2,3)
    trainDataInner.labels = data_set.trainLabels[1]
    testDataInner.data = data_set.test:transpose(1,4):transpose(2,3)
    testDataInner.labels = data_set.testLabels[1]

    -- crop patches
    print('cropping patch number ', opt.patchIndex)
    trainDataInner.data = DeepIdUtils.getPatch(trainDataInner.data, opt.patchIndex, useFlippedPatches)
    testDataInner.data = DeepIdUtils.getPatch(testDataInner.data, opt.patchIndex, useFlippedPatches)
    if useFlippedPatches then
        -- replicate labels
        trainDataInner.labels = torch.cat(trainDataInner.labels, trainDataInner.labels)
        testDataInner.labels = torch.cat(testDataInner.labels, testDataInner.labels)
        startFlipIndex = trainDataInner.data:size(1)/2 + 1 -- for visualization purposes
    end
else
    -- data is seperated into chunks
    print 'Loading data in chunks'
    setType = {'train', 'test' }
    for iSet = 1,2 do
        local iFile = 1
        local chunkFilePath
        local chunkData
        while true do
            chunkFilePath = dataFileName..'_'..setType[iSet]..'_'..iFile..'.'..fileFormat
            if paths.filep(chunkFilePath) then
                print('Loading from: ', chunkFilePath)
                chunkData = loadFunc(chunkFilePath)
                chunkData.data = chunkData.data:transpose(1,4):transpose(2,3)
                chunkData.labels = chunkData.labels[1]
                collectgarbage()

                -- crop patches
                print('cropping patch number ', opt.patchIndex)
                chunkData.data = DeepIdUtils.getPatch(chunkData.data, opt.patchIndex, useFlippedPatches)
                if useFlippedPatches then
                    -- replicate labels
                    chunkData.labels = torch.cat(chunkData.labels, chunkData.labels)
                end

                print(chunkData)

                -- concat data into relevant var
                if (setType[iSet] == 'train') then
                    if (trainDataInner.data:dim() == 0) then
                        trainDataInner.data = chunkData.data
                        trainDataInner.labels = chunkData.labels
                        startFlipIndex = chunkData.data:size(1)/2 + 1 -- for visualization purposes
                    else
                        trainDataInner.data = torch.cat(
                            trainDataInner.data, chunkData.data, 1)
                        trainDataInner.labels = torch.cat(trainDataInner.labels, chunkData.labels)
                    end
                elseif (setType[iSet] == 'test') then
                    if (testDataInner.data:dim() == 0) then
                        testDataInner.data = chunkData.data
                        testDataInner.labels = chunkData.labels
                    else
                        testDataInner.data = torch.cat(
                            testDataInner.data, chunkData.data, 1)
                        testDataInner.labels = torch.cat(testDataInner.labels, chunkData.labels)
                    end
                end
                iFile = iFile + 1
            else
                break
            end
        end
    end
end

if opt.trainOnly then
    trainDataInner = {
        data = torch.cat(trainDataInner.data,testDataInner.data, 1),
        labels = torch.cat(trainDataInner.labels, testDataInner.labels)
    }
    testDataInner = {}
end
trainDataInner.size = function() return trainDataInner.data:size(1) end
testDataInner.size = function() return testDataInner.data:size(1) end

print('train data :', trainDataInner)
print('test data :', testDataInner)

-- convert to our general dataset format
trainData = {
    numChunks = 1,
    getChunk = function(iChunk) return trainDataInner end
}
testData = {
    numChunks = 1,
    getChunk = function(iChunk) return testDataInner end
}

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

        print('samples: ', 1, nSamples)
        firstSamplesTrain[{{1, nSamples}}] = trainDataInner.data[{ {1,nSamples} }]

        print('flipped samples: ', startFlipIndex, startFlipIndex + nSamples - 1)
        firstSamplesTrain[{{nSamples+1, 2*nSamples}}] =
            trainDataInner.data[{ {startFlipIndex,startFlipIndex + nSamples - 1} }]
    else
        print('samples: ', 1, nSamples)
        firstSamplesTrain = trainDataInner.data[{ {1,nSamples} }]
    end

    gfx.image(firstSamplesTrain, {legend='train - samples'})

    --- gfx is buggy, so we will save images into files...
    require 'image'
    for iImage = 1,firstSamplesTrain:size(1) do
        im = firstSamplesTrain[iImage]
        image.save('visualize/'..opt.patchIndex..'_'..iImage..'.jpg',im)
    end
end