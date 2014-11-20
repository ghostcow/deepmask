package.path = package.path .. ";../nn/?.lua;"
require 'image'
require 'nn'
require 'cunn'
require 'ccn2'
require 'mattorch'
require 'options'

--- some ugly way to know if this file was loaded by require, or by command line run
-- similar to python check : if __name__ = "__main__":
local myname = ...
isCmd = true
if type(package.loaded[myname]) == "userdata" then
    isCmd = false
else
    --- first 2 input arguments are mandatory for this script
    inputPath = arg[1]  -- txt file with image paths
    outputPath = arg[2] -- output mat file path
    arg[1] = nil
    arg[2] = nil

    --- parsing left additional parameters
    opt = getOptions()
end
---

batchSize = opt.batchSize
imageDim = {152,152}

function getFaceFeatures(imagePaths, model, featureLayerIndex, mode)

    -- mode = 1 means the image paths are the keys of imagePaths,
    --        2 means imagePaths is an array with the paths as values
    --        when using mode=1 the output is a table with key=image path, value = feature
    --        when using mode=2 the output is a table with the features as elements
    if (mode == nil) then
        mode = 1
    end

    function processBatch(batchImagesPaths)
        inputs = torch.Tensor(batchSize, 3, imageDim[1], imageDim[2])
        iImage = 1
        isBadImage = {}
        for _,imagePath in pairs(batchImagesPaths) do
            if os.rename(imagePath, imagePath) then
                -- image file exist
                im = image.load(imagePath)
                isBadImage[imagePath] = false
            else
                -- image file doesn't exist (the alignment has failed)
                print(imagePath, ' - not found!')
                im = torch.Tensor(3, imageDim[1], imageDim[2]):fill(0)
                isBadImage[imagePath] = true
            end
            inputs[iImage] = im

            -- store the image index
            iImage = iImage + 1
        end

        collectgarbage() -- called to prevent cuda memory errors
        inputs = inputs:cuda()
        outputs = model:forward(inputs)
        features = model:get(featureLayerIndex).output:float()

        for iImage,imagePath in pairs(batchImagesPaths) do
            feature = features[iImage]
            if isBadImage[imagePath] then
                feature:fill(0)
            end

            if (mode == 1) then
                faceFeatures[imagePath] = feature
            elseif (mode == 2) then
                table.insert(faceFeatures, feature)
            end

        end
    end

    --- first of all, turn off all dropout modules
    for iModule = 1,#model.modules do
        if (torch.type(model.modules[iModule]) == 'nn.Dropout') then
            model.modules[iModule].train = false
        end
    end
    
    --- start processing
    faceFeatures = {}
    batchImagesPaths = {}
    if (mode == 1) then
        -- convert imagePaths into simple array
        imagePathsTemp = {}
        for imagePath,_ in pairs(imagePaths) do
            table.insert(imagePathsTemp, imagePath)
        end
        imagePaths = imagePathsTemp
    end

    for _,imagePath in pairs(imagePaths) do
        table.insert(batchImagesPaths, imagePath)
        if  (#batchImagesPaths == batchSize) then
            print('batch', #batchImagesPaths)
            processBatch(batchImagesPaths)
            batchImagesPaths = {}
        end
    end
    if (#batchImagesPaths > 0) then
        print('batch', #batchImagesPaths)
        processBatch(batchImagesPaths)
    end

    return faceFeatures
end

if isCmd then
    -- load image paths from input file
    imagePaths = {}
    for line in io.lines(inputPath) do
        table.insert(imagePaths, line)
    end

    local state_file_path = paths.concat('../results/', opt.save, 'model.net')
    model = torch.load(state_file_path)
    print(model)
    featureLayerIndex = #(model.modules) - 3 -- last 3 layers : dropout, fully conected, log
    faceFeatures = getFaceFeatures(imagePaths, model, featureLayerIndex, 2)
    mattorch.save(outputPath, faceFeatures)
end