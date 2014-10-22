LfwUtils = {}

LfwUtils.mainDir = '/media/data/datasets/LFW/lfw_aligned'
LfwUtils.pairsFilePath = '/media/data/datasets/LFW/view2/pairs.txt'

function LfwUtils.loadPairs()
    -- return table with image pairs & indication of match/mismatch for each pair
    local iFold = 1
    local iPair = 1
    pairsData = {}
    local currFoldMatchPairs = {}
    local currFoldMismatchPairs = {}
    local isInitialized = false
    for line in io.lines(LfwUtils.pairsFilePath) do
        if not isInitialized then
            -- first line
            local k = line:find('\t')
            LfwUtils.numFolds = tonumber(line:sub(1, k-1))
            LfwUtils.numPairsPerFold = tonumber(string.sub(line, k+1))
            isInitialized = true
        else
            -- ordinary line - one of 2 types : match/mismatch
            if (iPair <= LfwUtils.numPairsPerFold) then
                -- match pair
                local k = line:find('\t')
                local personName = line:sub(1, k-1)
                line = line:sub(k+1)
                k = line:find('\t')
                local imgNum1 = tonumber(line:sub(1, k-1))
                local imgNum2 = tonumber(line:sub(k+1))

                pair = {name=personName, imgNum1 = imgNum1, imgNum2 = imgNum2 }
                currFoldMatchPairs[iPair] = pair
            else
                -- mismatch pair
                pair = {}
                for i = 1,2 do
                    local k = line:find('\t')
                    local personName = line:sub(1, k-1)
                    local personImgNum
                    line = line:sub(k+1)
                    if (i==1) then
                        k = line:find('\t')
                        personImgNum = tonumber(line:sub(1, k-1))
                        line = line:sub(k+1)
                        pair["name1"] = personName
                        pair["imgNum1"] = personImgNum
                    else
                        personImgNum = tonumber(line)
                        pair["name2"] = personName
                        pair["imgNum2"] = personImgNum
                    end
                end
                currFoldMismatchPairs[iPair-LfwUtils.numPairsPerFold] = pair
            end

            iPair = iPair + 1
            if (iPair > 2*LfwUtils.numPairsPerFold) then
                -- save previous fold
                pairsData[iFold] = {match=currFoldMatchPairs, mismatch=currFoldMismatchPairs}
                currFoldMatchPairs = {}
                currFoldMismatchPairs = {}

                -- new fold indices
                iFold = iFold + 1
                iPair = 1
            end
        end
    end
    return pairsData
end

function LfwUtils.getImagePath(personName, personImgNum)
    imagePath = paths.concat(LfwUtils.mainDir, personName, string.format("%s_%04d.jpg", personName, personImgNum))
    return imagePath
end

function LfwUtils.getFoldPairsFeatures(fold, faceFeatures)
    foldSize = 2*LfwUtils.numPairsPerFold
    foldData = torch.FloatTensor(foldSize, 2, featureSize)
    foldLabels = torch.LongTensor(foldSize) -- +1 for match pair, -1 for mismatch
    local path1
    local path2
    local label
    for iPair = 1,2*LfwUtils.numPairsPerFold do
        if (iPair <= LfwUtils.numPairsPerFold) then
            -- match pair
            label = 1
            path1 = LfwUtils.getImagePath(fold.match[iPair].name, fold.match[iPair].imgNum1)
            path2 = LfwUtils.getImagePath(fold.match[iPair].name, fold.match[iPair].imgNum2)
        else
            -- mismatch pair
            label = -1
            path1 = LfwUtils.getImagePath(fold.mismatch[iPair-LfwUtils.numPairsPerFold].name1,
                fold.mismatch[iPair-LfwUtils.numPairsPerFold].imgNum1)
            path2 = LfwUtils.getImagePath(fold.mismatch[iPair-LfwUtils.numPairsPerFold].name2,
                fold.mismatch[iPair-LfwUtils.numPairsPerFold].imgNum2)
        end

        featurePair = torch.FloatTensor(2, featureSize)
        featurePair[{{1},{}}] = faceFeatures[path1]
        featurePair[{{2},{}}] = faceFeatures[path2]

        foldData[iPair] = featurePair
        foldLabels[iPair] = label
    end
    return foldData,foldLabels
end

function LfwUtils.getImagePaths(pairsData)
    --- to construct a set we use table where each key is image path (the value has no meaning)
    imagePaths = {}
    for iFold = 1,LfwUtils.numFolds do
        fold = pairsData[iFold]
        for iPair = 1,2*LfwUtils.numPairsPerFold do
            if (iPair <= LfwUtils.numPairsPerFold) then
                -- match pair
                path1 = LfwUtils.getImagePath(fold.match[iPair].name, fold.match[iPair].imgNum1)
                path2 = LfwUtils.getImagePath(fold.match[iPair].name, fold.match[iPair].imgNum2)
            else
                -- mismatch pair
                path1 = LfwUtils.getImagePath(fold.mismatch[iPair-LfwUtils.numPairsPerFold].name1,
                    fold.mismatch[iPair-LfwUtils.numPairsPerFold].imgNum1)
                path2 = LfwUtils.getImagePath(fold.mismatch[iPair-LfwUtils.numPairsPerFold].name2,
                    fold.mismatch[iPair-LfwUtils.numPairsPerFold].imgNum2)
            end
            imagePaths[path1] = true
            imagePaths[path2] = true
        end
    end
    return imagePaths
end

if DEBUG then
    --- tests
    x = LfwUtils.loadPairs()
    print('fold numbers :')
    for key,value in pairs(x) do
        print(key)
    end

    print('fold1 - match & mismatch pair:')
    fold1 = x[1]
    for key,value in pairs(fold1) do
        print(key)
        print(#value)
        print(value[1])
    end
end