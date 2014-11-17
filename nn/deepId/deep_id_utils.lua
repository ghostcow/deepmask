require 'image'
require 'paths'
function round(num) return math.floor(num+.5) end
currFileDir = paths.dirname(debug.getinfo(1,'S').source:sub(2))

DeepIdUtils = {}

DeepIdUtils.landmarksPath = paths.concat(currFileDir, '../../img_preproc/landmarks_target.txt')
-- currently we support only one scale & square rgb patch
-- TODO : support 15 different patches for 3 scales
-- scale = 1 : 5 31x31, 4 31x39, 1 ?x47
-- scale = 2 : 5 45x45, 4 45x55 , 1 ?x45 (not sure about 55 &
-- scale = 3 : 5 63x63, 4 63x79, 1 ?x63
DeepIdUtils.numScales = 1
DeepIdUtils.inputModes = {'rgb' }
DeepIdUtils.numPatches = 5 -- left eye, right eye, nose, left lip, right lip
DeepIdUtils.patchCenters = torch.Tensor(2, DeepIdUtils.numPatches)
DeepIdUtils.patchSizeOriginal = 81
DeepIdUtils.patchSizeTarget = 31
DeepIdUtils.imageDim = {152,152}

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

-- patch no. 1 : left eye
DeepIdUtils.patchCenters[{{},{1}}] = (DeepIdUtils.landmarksLocs[{{},{1}}] + DeepIdUtils.landmarksLocs[{{},{2}}]) / 2
-- patch no. 2 : right eye
DeepIdUtils.patchCenters[{{},{2}}] = (DeepIdUtils.landmarksLocs[{{},{3}}] + DeepIdUtils.landmarksLocs[{{},{4}}]) / 2
-- patch no. 3 : nose center
DeepIdUtils.patchCenters[{{},{3}}] = (DeepIdUtils.landmarksLocs[{{},{5}}] + DeepIdUtils.landmarksLocs[{{},{7}}]) / 2
-- patch no. 4 : left lip edge
DeepIdUtils.patchCenters[{{},{4}}] = DeepIdUtils.landmarksLocs[{{},{8}}]
-- patch no. 5 : right lip edge
DeepIdUtils.patchCenters[{{},{5}}] = DeepIdUtils.landmarksLocs[{{},{9}}]
-- map each index to the one that has to be flipped in order to get more data
DeepIdUtils.patchFlippedIndices = {2,1,3,5,4}

DeepIdUtils.patchBordes = torch.Tensor(4, DeepIdUtils.numPatches)
for iPatch = 1,DeepIdUtils.numPatches do
    center = DeepIdUtils.patchCenters[{{},iPatch}]

    local patchRadius = (DeepIdUtils.patchSizeOriginal - 1) / 2
    x0 = round(center[1]) - patchRadius
    y0 = round(center[2]) - patchRadius

    x1 = x0 + DeepIdUtils.patchSizeOriginal - 1
    y1 = y0 + DeepIdUtils.patchSizeOriginal - 1

    if (x0 < 1) then
        x0 = 1
        x1 = x1 + DeepIdUtils.patchSizeOriginal - 1
    end
    if (x1 > DeepIdUtils.imageDim[2]) then
        x1 = DeepIdUtils.imageDim[2]
        x0 = x1 - DeepIdUtils.patchSizeOriginal + 1
    end
    if (y0 < 1) then
        y0 = 1
        y1 = y1 + DeepIdUtils.patchSizeOriginal - 1
    end
    if (y1 > DeepIdUtils.imageDim[1]) then
        y1 = DeepIdUtils.imageDim[1]
        y0 = y1 - DeepIdUtils.patchSizeOriginal + 1
    end

    DeepIdUtils.patchBordes[{{},{iPatch}}] = torch.Tensor({y0,y1,x0,x1})
end

function DeepIdUtils.getPatch(images, patchIndex, useFlipped)
    -- images : tensor of format nImages x 3 x height x width
    -- patchIndex : indicating the requested patch (1-5)

    patchBorders = DeepIdUtils.patchBordes[{{},patchIndex}]

    -- crop patches centered in center
    patches = images[{{},{},{patchBorders[1], patchBorders[2]},
        {patchBorders[3], patchBorders[4]}}]

    -- resize patches to DeepIdUtils.patchSizeTarget
    nImages = images:size(1)
    nChannels = images:size(2)
    local patchesResized = torch.Tensor(nImages, nChannels, DeepIdUtils.patchSizeTarget, DeepIdUtils.patchSizeTarget)
    for iImage = 1,nImages do
        patchesResized[iImage] = image.scale(patches[iImage], DeepIdUtils.patchSizeTarget)
    end

    local flipeedPatchResized
    local patchesResizedTotal
    -- data augmentation : using flipped patches also
    if useFlipped then
        patchFlippedIndex = DeepIdUtils.patchFlippedIndices[patchIndex]
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
        patchesResizedTotal = torch.Tensor(2*nImages, nChannels, DeepIdUtils.patchSizeTarget, DeepIdUtils.patchSizeTarget)
        patchesResizedTotal[{{1, nImages}}] = patchesResized
        patchesResizedTotal[{{nImages+1, 2*nImages}}] = flipeedPatchResized
    else
        patchesResizedTotal = patchesResized
    end

    return patchesResizedTotal
end
