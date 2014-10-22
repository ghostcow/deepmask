require 'image'

batchSize = 128
imageDim = 152

function getFaceFeatures(imagePaths, model, featureLayerIndex, normFactors)
    function processBatch(batchImagesPaths)
        inputs = torch.Tensor(batchSize, 3, imageDim, imageDim)
        iImage = 1
        for _,imagePath in pairs(batchImagesPaths) do
            if os.rename(imagePath, imagePath) then
                -- image file exist
                im = image.load(imagePath)
            else
                -- image file doesn't exist (the alignment has failed)
                print(imagePath, ' - not found!')
                im = torch.Tensor(3, imageDim, imageDim):fill(0)
            end

            inputs[iImage] = im

            -- store the image index
            faceFeatures[imagePath] = iImage
            iImage = iImage + 1
        end

        collectgarbage() -- called to prevent cude memory errors
        inputs = inputs:cuda()
        outputs = model:forward(inputs)
        features = model:get(featureLayerIndex).output:float()
        if normFactors then
            -- TODO : normalize features
        end

        for _,imagePath in pairs(batchImagesPaths) do
            iImage = faceFeatures[imagePath]
            feature = features[iImage]
            faceFeatures[imagePath] = feature
        end
    end

    faceFeatures = {}
    batchImagesPaths = {}
    for imagePath,_ in pairs(imagePaths) do
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