require 'image'

batchSize = 128
imageDim = 152

function getFaceFeatures(imagePaths, model, featureLayerIndex, mode)

    -- mode = 1 means the image paths are the keys of imagePaths,
    --        2 means imagePaths is an array with the paths as values
    --        when using mode=1 the output is a table with key=image path, value = feature
    --        when using mode=2 the output is a table with the features as elements
    if (mode == nil) then
        mode = 1
    end

    function processBatch(batchImagesPaths)
        inputs = torch.Tensor(batchSize, 3, imageDim, imageDim)
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
                im = torch.Tensor(3, imageDim, imageDim):fill(0)
                isBadImage[imagePath] = true
            end
            inputs[iImage] = im

            -- store the image index
            iImage = iImage + 1
        end

        collectgarbage() -- called to prevent cude memory errors
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