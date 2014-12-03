require 'image'
require 'paths'
require 'math'
package.path = package.path .. ";../?.lua"
require 'options'
currFileDir = paths.dirname(debug.getinfo(1,'S').source:sub(2))

-- parse command line arguments
if not opt then
    print '==> processing options'
    opt = getOptions()
end

DeepIdUtils = {}

-- currently we support only one scale & square rgb patch
-- support 3 scales x 10 patches, each scale has 3 patched types : square,frame,profile
-- scale = 1 : 5 31x31, 4 31x39 (scaled to 39x39), 1 47x39 (scaled to 47x47)
-- scale = 2 : 5 45x45, 4 45x53 (scaled to 53x53), 1 61x53 (scaled to 61x61)
-- scale = 3 : 5 59x59, 4 59x67 (scaled to 67x67), 1 75x67 (scaled to 75x75)
DeepIdUtils.inputModes = {'rgb' }
DeepIdUtils.numPatches = 5 -- #patches per scale : left eye, right eye, nose, left lip, right lip

if (opt.deepIdMode == 1) then
    -- mode 1 : using the images aligned for deepface (size=152x152)
    DeepIdUtils.numScales = 1
        -- patch sizes for each sacle
    DeepIdUtils.patchSizeOriginal = {81}
    DeepIdUtils.patchSizeTarget = {31}
    DeepIdUtils.imageDim = {152,152}
    DeepIdUtils.landmarksPath = paths.concat(currFileDir, '../../img_preproc/landmarks_aligned_deepface.txt')
elseif (opt.deepIdMode == 2) then
    -- mode 2 : using the images aligned for deepid (size=140x115, more background around face)
    DeepIdUtils.imageDim = {140,115}
    DeepIdUtils.numScales = 3
    DeepIdUtils.numPatches = 10 -- (5 square, 4 frame, 1 profile)
    DeepIdUtils.numSquarePatches = 5
    DeepIdUtils.numFramePatches = 4
    DeepIdUtils.numProfilePatches = 1
    
    -- patch sizes for each sacle
    -- square patches, frame patches (horizontal frames, scaled to square images)
    -- profile patches (vertical frames, scaled to square images)
    -- {width, height}
    DeepIdUtils.patchSizeOriginal = {31, 45, 59, {31,39}, {45,53}, {59,67}, {47,39}, {61,53}, {75,67}} 
    DeepIdUtils.patchSizeTarget = {31, 45, 59, {39,39}, {53,53}, {67,67}, {47,47}, {61,61}, {75,75}}
    -- DeepIdUtils.patchSizeTarget = {31, 45, 59, {31,39}, {45,53}, {59,67}, {47,39}, {61,53}, {75,67}}
    DeepIdUtils.landmarksPath = paths.concat(currFileDir, '../../img_preproc/landmarks_aligned_deepid.txt')
end

DeepIdUtils.patchCenters = torch.Tensor(2, DeepIdUtils.numPatches)

DeepIdUtils.maxIndexSquarePatches = (DeepIdUtils.numScales - 1)*DeepIdUtils.numSquarePatches + DeepIdUtils.numSquarePatches
DeepIdUtils.maxIndexFramePatches = DeepIdUtils.maxIndexSquarePatches + (DeepIdUtils.numScales - 1)*DeepIdUtils.numFramePatches + (DeepIdUtils.numFramePatches)

function DeepIdUtils.getPatchIndex(iPatch, iScale)
    -- 1:15  - 5 square patches, scale 1-3
    -- 16:27 - 4 frame patcehs, scale 1-3
    -- 28:30 - profile patch, scale 1-3
    if (iPatch <= DeepIdUtils.numSquarePatches) then
      return (iScale - 1)*DeepIdUtils.numSquarePatches + iPatch
    elseif (iPatch <= (DeepIdUtils.numSquarePatches + DeepIdUtils.numFramePatches)) then
      return (iScale - 1)*DeepIdUtils.numFramePatches + (iPatch - DeepIdUtils.numSquarePatches) +
              DeepIdUtils.maxIndexSquarePatches
    elseif (iPatch == (DeepIdUtils.numSquarePatches + DeepIdUtils.numFramePatches + DeepIdUtils.numProfilePatches)) then
      return (iScale - 1)*DeepIdUtils.numProfilePatches + (iPatch - DeepIdUtils.numSquarePatches - DeepIdUtils.numFramePatches) +
              DeepIdUtils.maxIndexFramePatches
    end
end
function DeepIdUtils.parsePatchIndex(patchIndex)
    -- convert patchIndex to iPatch & iScale
    if (patchIndex > DeepIdUtils.maxIndexSquarePatches) then
        if (patchIndex <= DeepIdUtils.maxIndexFramePatches) then
            -- frame patch
            patchIndex = patchIndex - DeepIdUtils.maxIndexSquarePatches
            numPatchesType = DeepIdUtils.numFramePatches
            iPatchOffset = DeepIdUtils.numSquarePatches
            iType = 2
        else
            -- profile patch
            patchIndex = patchIndex - DeepIdUtils.maxIndexFramePatches
            numPatchesType = DeepIdUtils.numProfilePatches
            iPatchOffset = DeepIdUtils.numSquarePatches + DeepIdUtils.numFramePatches
            iType = 3
        end
    else
      iPatchOffset = 0
      numPatchesType = DeepIdUtils.numSquarePatches
      iType = 1
    end
    
    iScale = 1 + math.floor((patchIndex-1)/numPatchesType)
    iPatch = 1 + ((patchIndex-1) % numPatchesType) + iPatchOffset
    return iPatch,iScale,iType
end

-- parse landmarks file
DeepIdUtils.landmarksLocs = torch.Tensor(2, 9)
local iLine = 1
for line in io.lines(DeepIdUtils.landmarksPath) do
    local iCoord = 1
    for word in string.gmatch(line, '([^,]+)') do
        DeepIdUtils.landmarksLocs[{{iLine},{iCoord}}] = tonumber(word)
        iCoord = iCoord + 1
    end
    iLine = iLine + 1
end

--- Square patches
-- patch no. 1 : left eye
DeepIdUtils.patchCenters[{{},{1}}] = (DeepIdUtils.landmarksLocs[{{},{1}}] + DeepIdUtils.landmarksLocs[{{},{2}}]) / 2
-- patch no. 2 : right eye
DeepIdUtils.patchCenters[{{},{2}}] = (DeepIdUtils.landmarksLocs[{{},{3}}] + DeepIdUtils.landmarksLocs[{{},{4}}]) / 2
-- patch no. 3 : nose center
nose = (DeepIdUtils.landmarksLocs[{{},{5}}] + DeepIdUtils.landmarksLocs[{{},{7}}]) / 2
DeepIdUtils.patchCenters[{{},{3}}] = nose
-- patch no. 4 : left lip edge
DeepIdUtils.patchCenters[{{},{4}}] = DeepIdUtils.landmarksLocs[{{},{8}}]
-- patch no. 5 : right lip edge
DeepIdUtils.patchCenters[{{},{5}}] = DeepIdUtils.landmarksLocs[{{},{9}}]
--- Frames patches
eyesCenter = (DeepIdUtils.patchCenters[{{},{1}}] + DeepIdUtils.patchCenters[{{},{2}}])/2
DeepIdUtils.patchCenters[{{},{6}}] = eyesCenter - torch.Tensor({0,15})
DeepIdUtils.patchCenters[{{},{7}}] = eyesCenter
DeepIdUtils.patchCenters[{{},{8}}] = torch.Tensor({DeepIdUtils.patchCenters[{1,6}], (eyesCenter[{2,1}]+nose[{2,1}])/2})
DeepIdUtils.patchCenters[{{},{9}}] = torch.Tensor({DeepIdUtils.patchCenters[{1,6}], nose[{2,1}]})
--- Profile patch
DeepIdUtils.patchCenters[{{},{10}}] = (eyesCenter+nose)/2

-- map each index to the one that has to be flipped in order to get augmented data
DeepIdUtils.patchFlippedIndices = {2,1,3,5,4,6,7,8,9,10}

DeepIdUtils.patchBordes = torch.Tensor(4, DeepIdUtils.numScales*DeepIdUtils.numPatches)
for iScale = 1,DeepIdUtils.numScales do
    for iPatch = 1,DeepIdUtils.numPatches do
        if (iPatch <= DeepIdUtils.numSquarePatches) then
            iType = 1
        elseif (iPatch <= (DeepIdUtils.numSquarePatches + DeepIdUtils.numFramePatches)) then
            iType = 2
        else
            iType = 3
        end
        local center = DeepIdUtils.patchCenters[{{},iPatch}]
        local patchSize = DeepIdUtils.patchSizeOriginal[(iType - 1)*DeepIdUtils.numScales + iScale]
        if (type(patchSize) == 'number') then
          patchSize = {patchSize, patchSize}
        end
        patchSize = torch.Tensor(patchSize)
	
        local patchRadius = (patchSize - 1) / 2
        local topLeft = torch.round(center - patchRadius)
        local bottomRight = topLeft + patchSize - 1
        
        if (topLeft[1] < 1) then
            topLeft[1] = 1
            bottomRight[1] = bottomRight[1] + patchSize[1] - 1
        end
        if (bottomRight[1] > DeepIdUtils.imageDim[2]) then
            bottomRight[1] = DeepIdUtils.imageDim[2]
            topLeft[1] = bottomRight[1] - patchSize[1] + 1
        end
        if (topLeft[2] < 1) then
            topLeft[2] = 1
            bottomRight[2] = bottomRight[2] + patchSize - 1
        end
        if (bottomRight[2] > DeepIdUtils.imageDim[1]) then
            bottomRight[2] = DeepIdUtils.imageDim[1]
            topLeft[2] = bottomRight[2] - patchSize + 1
        end

        patchIndex = DeepIdUtils.getPatchIndex(iPatch, iScale)
        DeepIdUtils.patchBordes[{{},{patchIndex}}] = torch.Tensor({topLeft[2],bottomRight[2],topLeft[1],bottomRight[1]})
    end
end

function DeepIdUtils.getPatch(images, patchIndex, useFlipped)
    -- images : tensor of format nImages x 3 x height x width
    -- patchIndex : indicating the requested patch (1-5)

    patchBorders = DeepIdUtils.patchBordes[{{},patchIndex}]
    iPatch,iScale,iType = DeepIdUtils.parsePatchIndex(patchIndex)
    patchSizeTarget = DeepIdUtils.patchSizeTarget[(iType - 1)*DeepIdUtils.numScales + iScale]
    print('DeepIdUtils.getPatch : index=', patchIndex, 'iPatch=', iPatch, 'iScale=', iScale, 'iType=', iType)
    print('patch size = ', patchSizeTarget)

    -- crop patches centered in center
    patches = images[{{},{},{patchBorders[1], patchBorders[2]},
        {patchBorders[3], patchBorders[4]}}]

    -- resize patches to DeepIdUtils.patchSizeTarget
    nImages = images:size(1)
    nChannels = images:size(2)

    if (type(patchSizeTarget) == 'number') then
      patchSizeTarget = {patchSizeTarget, patchSizeTarget}
    end
    local patchesResized = torch.Tensor(nImages, nChannels,
        patchSizeTarget[2], patchSizeTarget[1])
    for iImage = 1,nImages do
        patchesResized[iImage] = image.scale(patches[iImage], patchSizeTarget[2], patchSizeTarget[1])
    end

    local flipeedPatchResized
    local patchesResizedTotal
    -- data augmentation : using flipped patches also
    if useFlipped then
        patchFlippedIndex = DeepIdUtils.getPatchIndex(DeepIdUtils.patchFlippedIndices[iPatch], iScale)
        if (patchFlippedIndex == patchIndex) then
            -- using the same patch
            flipeedPatchResized = patchesResized:clone()
        else
            flipeedPatchResized = DeepIdUtils.getPatch(images, patchFlippedIndex, false)
        end
        -- no we will do the actual flipping
        for iImage = 1,nImages do
            flipeedPatchResized[iImage] = image.hflip(flipeedPatchResized[iImage])
        end

        -- fill into the output tensor
        patchesResizedTotal = torch.Tensor(2*nImages, nChannels,
            patchSizeTarget[2], patchSizeTarget[1])
        patchesResizedTotal[{{1, nImages}}] = patchesResized
        patchesResizedTotal[{{nImages+1, 2*nImages}}] = flipeedPatchResized
    else
        patchesResizedTotal = patchesResized
    end

    return patchesResizedTotal
end