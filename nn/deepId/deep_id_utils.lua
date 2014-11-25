require 'image'
require 'paths'
require 'math'
function round(num) return math.floor(num+.5) end
currFileDir = paths.dirname(debug.getinfo(1,'S').source:sub(2))

-- parse command line arguments
if not opt then
    print '==> processing options'
    opt = getOptions()
end

DeepIdUtils = {}

-- currently we support only one scale & square rgb patch
-- TODO : support 3 scales x 10 patches (currently only 3 x 5 = 15 are supported)
-- scale = 1 : 5 31x31, 4 31x39, 1 ?x47
-- scale = 2 : 5 45x45, 4 45x55 , 1 ?x45
-- scale = 3 : 5 63x63, 4 63x79, 1 ?x63
DeepIdUtils.inputModes = {'rgb' }
DeepIdUtils.numPatches = 5 -- #patches per scale : left eye, right eye, nose, left lip, right lip
DeepIdUtils.patchCenters = torch.Tensor(2, DeepIdUtils.numPatches)

function DeepIdUtils.getPatchIndex(iPatch, iScale)
    return (iScale - 1)*DeepIdUtils.numPatches + iPatch
end
function DeepIdUtils.parsePatchIndex(patchIndex)
    -- convert patchIndex to iPatch & iScale
    iScale = 1 + math.floor((patchIndex-1)/DeepIdUtils.numPatches)
    iPatch = 1 + (patchIndex-1) % DeepIdUtils.numPatches
    return iPatch,iScale
end

if (opt.deepIdMode == 1) then
    -- mode 1 : using the images aligned for deepface (size=152x152)
    DeepIdUtils.numScales = 1
        -- patch sizes for each sacle
    DeepIdUtils.patchSizeOriginal = {81}
    DeepIdUtils.patchSizeTarget = {31}
    DeepIdUtils.imageDim = {152,152}
    DeepIdUtils.landmarksPath = paths.concat(currFileDir, '../../img_preproc/landmarks_aligned_deepface.txt')
elseif (opt.deepIdMode == 2) then
    -- mode 2 : using the images aligned for deepid (size=279x230, more background around face)
    DeepIdUtils.numScales = 3
    -- patch sizes for each sacle
    DeepIdUtils.patchSizeOriginal = {31, 45, 59} --{71, 91, 111}
    DeepIdUtils.patchSizeTarget = {31, 45, 59}
    DeepIdUtils.imageDim = {140,115} --{279,230}
    DeepIdUtils.landmarksPath = paths.concat(currFileDir, '../../img_preproc/landmarks_aligned_deepid.txt')
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

DeepIdUtils.patchBordes = torch.Tensor(4, DeepIdUtils.numScales*DeepIdUtils.numPatches)
for iScale = 1,DeepIdUtils.numScales do
    for iPatch = 1,DeepIdUtils.numPatches do
        local center = DeepIdUtils.patchCenters[{{},iPatch}]
        local patchSize = DeepIdUtils.patchSizeOriginal[iScale]
        local patchRadius = (patchSize - 1) / 2

        x0 = round(center[1]) - patchRadius
        y0 = round(center[2]) - patchRadius

        x1 = x0 + patchSize - 1
        y1 = y0 + patchSize - 1

        if (x0 < 1) then
            x0 = 1
            x1 = x1 + patchSize - 1
        end
        if (x1 > DeepIdUtils.imageDim[2]) then
            x1 = DeepIdUtils.imageDim[2]
            x0 = x1 - patchSize + 1
        end
        if (y0 < 1) then
            y0 = 1
            y1 = y1 + patchSize - 1
        end
        if (y1 > DeepIdUtils.imageDim[1]) then
            y1 = DeepIdUtils.imageDim[1]
            y0 = y1 - patchSize + 1
        end

        patchIndex = DeepIdUtils.getPatchIndex(iPatch, iScale)
        DeepIdUtils.patchBordes[{{},{patchIndex}}] = torch.Tensor({y0,y1,x0,x1})
    end
end

function DeepIdUtils.getPatch(images, patchIndex, useFlipped)
    -- images : tensor of format nImages x 3 x height x width
    -- patchIndex : indicating the requested patch (1-5)

    patchBorders = DeepIdUtils.patchBordes[{{},patchIndex}]
    iPatch,iScale = DeepIdUtils.parsePatchIndex(patchIndex)
    print('DeepIdUtils.getPatch :', patchIndex, iPatch, iScale)
    print('patch size = ', DeepIdUtils.patchSizeTarget[iScale])

    -- crop patches centered in center
    patches = images[{{},{},{patchBorders[1], patchBorders[2]},
        {patchBorders[3], patchBorders[4]}}]

    -- resize patches to DeepIdUtils.patchSizeTarget
    nImages = images:size(1)
    nChannels = images:size(2)
    local patchesResized = torch.Tensor(nImages, nChannels,
        DeepIdUtils.patchSizeTarget[iScale], DeepIdUtils.patchSizeTarget[iScale])
    for iImage = 1,nImages do
        patchesResized[iImage] = image.scale(patches[iImage], DeepIdUtils.patchSizeTarget[iScale])
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
            DeepIdUtils.patchSizeTarget[iScale], DeepIdUtils.patchSizeTarget[iScale])
        patchesResizedTotal[{{1, nImages}}] = patchesResized
        patchesResizedTotal[{{nImages+1, 2*nImages}}] = flipeedPatchResized
    else
        patchesResizedTotal = patchesResized
    end

    return patchesResizedTotal
end