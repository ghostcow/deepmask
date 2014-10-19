require 'image'
require 'options'

opt = getOptions()
imageDim = 152
batchSize = opt.batchSize

dataDir = '../data_set/clean_aligned_faces/data'
inputFilePath = paths.concat(dataDir, 'images_unknown.txt')
outputFileName = paths.concat(dataDir, 'images_unknown_results.txt')

function processBatch(fid, imagePaths)
    -- load images and feed into the network
    nImages = table.getn(imagePaths)
    inputs = torch.Tensor(batchSize, 3, imageDim, imageDim):cuda()
    for iImage = 1,nImages do
        imPath = imagePaths[iImage]
        im = image.load(imPath):cuda()
        inputs[iImage] = im
    end
    outputs = model:forward(inputs)
    for iImage = 1,nImages do
        imPath = imagePaths[iImage]
        result = outputs[iImage]
        fid:write(imPath,',',tostring(result[1]),',',tostring(result[2]),'\n')
    end
end


batchImagesPaths = {}
fid = io.open(outputFileName, "w")
for imPath in io.lines(inputFilePath) do
    table.insert(batchImagesPaths, line)
    if  table.getn(batchImagesPaths) == batchSize then
        -- process batch
        processBatch(batchImagesPaths)
    end
    batchImagesPaths = {}
end
if  (table.getn(batchImagesPaths) < batchSize) then
    processBatch(batchImagesPaths)
end
fid:close()
