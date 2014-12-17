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

function cropPatches(data, labels, useFlippedPatches)
    dataInner = {}
    -- the original matlab format is nImages x 3 x height x width
    -- (where height=width=152)
    -- but it's loaded into torch like this : width x height x 3 x nImages
    dataInner.data = data:transpose(1,4):transpose(2,3)
    dataInner.labels = labels[1]
    -- crop patches
    dataInner.data = DeepIdUtils.getPatch(dataInner.data, opt.patchIndex, useFlippedPatches)
    if useFlippedPatches then
        -- replicate labels
        dataInner.labels = torch.cat(dataInner.labels, dataInner.labels)
    end
    dataInner.size = function() return dataInner.data:size(1) end
    return dataInner
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

nLabels = 0
print('cropping patch number ', opt.patchIndex)
if paths.filep(dataFilePath) then
    --- all data in one file
    print('Loading from: ', dataFilePath)
    data_set = loadFunc(dataFilePath)
    trainDataInner = cropPatches(data_set.train, data_set.trainLabels, useFlippedPatches)
    testDataInner = cropPatches(data_set.test, data_set.testLabels, useFlippedPatches)
    nLabels = trainDataInner.labels:max()
    print('train data :', trainDataInner)
    print('test data :', testDataInner)
else
    --- data is seperated into chunks (if opt.useDatasetChunks=false we will load all chunks into memory)
    print 'Loading data in chunks'
    setType = {'train', 'test' }
    for iSet = 1,2 do
        local iFile = 1
        local chunkFilePath
        local chunkData
        dataInner = {
            data = torch.Tensor(),
            labels = torch.Tensor() }
        chunksPaths = {}
        collectgarbage()
        while true do
            chunkFilePath = dataFileName..'_'..setType[iSet]..'_'..iFile..'.'..fileFormat
            if not paths.filep(chunkFilePath) then
                break
            end

            print('Loading from: ', chunkFilePath)
            local time = sys.clock()
            chunkData = loadFunc(chunkFilePath)
            print('Loading time= '..(sys.clock() - time)..' [s]')
            time = sys.clock()
            chunkData = cropPatches(chunkData.data, chunkData.labels, useFlippedPatches)
            print('Cropping time= '..(sys.clock() - time)..' [s]')
            print(chunkData)

            nLabels = math.max(nLabels, chunkData.labels:max())
            if not opt.useDatasetChunks then
                -- concat data into relevant var
                if (dataInner.data:dim() == 0) then
                    dataInner.data = chunkData.data
                    dataInner.labels = chunkData.labels
                else
                    dataInner.data = torch.cat(
                        dataInner.data, chunkData.data, 1)
                    dataInner.labels = torch.cat(dataInner.labels, chunkData.labels)
                end
            end
            table.insert(chunksPaths, chunkFilePath)
            iFile = iFile + 1
        end
        if not opt.useDatasetChunks then
            -- all chunks will be loaded into memory
            if (setType[iSet] == 'train') then
                trainDataInner = {data = dataInner.data:clone(), labels = dataInner.labels:clone()}
            else
                testDataInner = {data = dataInner.data:clone(), labels = dataInner.labels:clone()}
            end
        else
            if (setType[iSet] == 'train') then
                trainData = {chunkPaths = chunksPaths}
            else
                testData = {chunkPaths = chunksPaths}
            end
        end
    end
end

if trainDataInner then
    -- all dataset was loaded into memory
    trainDataInner.size = function() return trainDataInner.data:size(1) end
    testDataInner.size = function() return testDataInner.data:size(1) end

    -- convert to our general dataset format
    trainData = {
        numChunks = 1,
        getChunk = function(iChunk) return trainDataInner end
    }
    testData = {
        numChunks = 1,
        getChunk = function(iChunk) return testDataInner end
    }
else
    function readChunkPatches(chunkFilePath)
        chunkData = loadFunc(chunkFilePath)
        chunkData = cropPatches(chunkData.data, chunkData.labels, useFlippedPatches)
        return chunkData
    end

    -- each dataset chunk will be loaded when needed
    trainData.numChunks = #trainData.chunkPaths
    trainData.getChunk = function(iChunk) return readChunkPatches(trainData.chunkPaths[iChunk]) end

    testData.numChunks = #testData.chunkPaths
    testData.getChunk = function(iChunk) return readChunkPatches(testData.chunkPaths[iChunk]) end
end

print('train data :', trainData)
print('test data :', testData)

-- classes - define classes array (used later for computing confusion matrix)
classes = {}
for i=1,nLabels do
    table.insert(classes, tostring(i))
end

------visualizing data---------------------------
if opt.visualize then
    nSamples = 100
    require 'gfx.js'
    print '==> visualizing data'
    local chunk = trainData.getChunk(1)
    if (false) then
        require 'gnuplot'
        gnuplot.figure(1)
        gnuplot.hist(chunk.labels, chunk.labels:max())
        gnuplot.title('#samples per label - training')
    end

    local firstSamplesTrain
    print('samples: ', 1, nSamples)
    firstSamplesTrain = chunk.data[{ {1,nSamples} }]

    gfx.image(firstSamplesTrain, {legend='train - samples'})

    --- gfx is buggy, so we will save images into files...
    require 'image'
    for iImage = 1,firstSamplesTrain:size(1) do
        im = firstSamplesTrain[iImage]
        image.save('visualize/'..opt.patchIndex..'_'..iImage..'.jpg',im)
    end
end