require 'image'
require 'paths'
require 'math'
package.path = package.path .. ";../?.lua"
require 'options'

DeepId2Utils = {}
DeepId2Utils.verbose = false
DeepId2Utils.imageDim = {140,115} -- {height, width}
DeepId2Utils.imageCenter = {57, 69.5} -- {x, y}

DeepId2Utils.patchSizeOriginal = {}
DeepId2Utils.patchSizeTarget = {}
DeepId2Utils.isRgb = {}
DeepId2Utils.isFlipped = {}
DeepId2Utils.numPatches = 25
DeepId2Utils.patchCenters = torch.Tensor(2, DeepId2Utils.numPatches)

--- All patches definitions (center, size, isFlipped & isRgb)

iPatch = 1
local patchSizeTarget = 45
local patchSizeOriginal = 81
local isRgb     = {false,false,false,false,true}
local isFlipped = {false,true,false,false,false }
local centers = {{52,87},{68,100},{41,59},{58,77},{58,95}}
for k = 1,5 do
    DeepId2Utils.patchSizeOriginal[iPatch] = patchSizeOriginal
    DeepId2Utils.patchSizeTarget[iPatch] = patchSizeTarget
    DeepId2Utils.isRgb[iPatch] = isRgb[k]
    DeepId2Utils.isFlipped[iPatch] = isFlipped[k]
    DeepId2Utils.patchCenters[{{},{iPatch}}] = torch.Tensor(centers[k])
    iPatch = iPatch + 1
end

patchSizeOriginal = 59
isRgb = {true,true,false }
isFlipped = {false,true,true }
centers = {{52,84},{52,75},{56,89} }
for k = 1,3 do
    DeepId2Utils.patchSizeOriginal[iPatch] = patchSizeOriginal
    DeepId2Utils.patchSizeTarget[iPatch] = patchSizeTarget
    DeepId2Utils.isRgb[iPatch] = isRgb[k]
    DeepId2Utils.isFlipped[iPatch] = isFlipped[k]
    DeepId2Utils.patchCenters[{{},{iPatch}}] = torch.Tensor(centers[k])
    iPatch = iPatch + 1
end

patchSizeOriginal = 45
isRgb = {false,false,false,false,false,false,
         true,true,true,true,true,true}
isFlipped = {false,true,true,false,false,true,
             false,true,false,true,false,true}
centers = {{58,107},{65,101},{53,88},{53,58},{36,67},{69,67},
           {44,102},{64,84},{64,81},{68,58},{41,66},{89,66}}
for k = 1,12 do
    DeepId2Utils.patchSizeOriginal[iPatch] = patchSizeOriginal
    DeepId2Utils.patchSizeTarget[iPatch] = patchSizeTarget
    DeepId2Utils.isRgb[iPatch] = isRgb[k]
    DeepId2Utils.isFlipped[iPatch] = isFlipped[k]
    DeepId2Utils.patchCenters[{{},{iPatch}}] = torch.Tensor(centers[k])
    iPatch = iPatch + 1
end

DeepId2Utils.patchSizeOriginal[iPatch] = {91,111}
DeepId2Utils.patchSizeTarget[iPatch] = 59
DeepId2Utils.isRgb[iPatch] = true
DeepId2Utils.isFlipped[iPatch] = false
DeepId2Utils.patchCenters[{{},{iPatch}}] = torch.Tensor({1+(115-1)/2,68})
iPatch = iPatch + 1

DeepId2Utils.patchSizeOriginal[iPatch] = {111,139}
DeepId2Utils.patchSizeTarget[iPatch] = 59
DeepId2Utils.isRgb[iPatch] = true
DeepId2Utils.isFlipped[iPatch] = false
DeepId2Utils.patchCenters[{{},{iPatch}}] = torch.Tensor({1+(115-1)/2,70})
iPatch = iPatch + 1

patchSizeTarget = 45
patchSizeOriginal = {107,85}
isRgb = {true,false,false}
isFlipped = {false,true,true}
centers = {{60,89},{60,75},{60,45}}
for k = 1,3 do
    DeepId2Utils.patchSizeOriginal[iPatch] = patchSizeOriginal
    DeepId2Utils.patchSizeTarget[iPatch] = patchSizeTarget
    DeepId2Utils.isRgb[iPatch] = isRgb[k]
    DeepId2Utils.isFlipped[iPatch] = isFlipped[k]
    DeepId2Utils.patchCenters[{{},{iPatch}}] = torch.Tensor(centers[k])
    iPatch = iPatch + 1
end
---

-- pre-calculate all patches borders
DeepId2Utils.patchBordes = torch.Tensor(4, DeepId2Utils.numPatches)
DeepId2Utils.patchBordesFlipped = torch.Tensor(4, DeepId2Utils.numPatches)
if DeepId2Utils.verbose then
    print(DeepId2Utils)
end

for iPatch = 1, DeepId2Utils.numPatches do
	local patchSize =  DeepId2Utils.patchSizeOriginal[iPatch]
	if (type(patchSize) == 'number') then
	  patchSize = {patchSize, patchSize}
    end

	patchSize = torch.Tensor(patchSize)
	
	for k =1,2 do
		-- 1st iteration - original patch, 2nd - flipped patch
		local center =  DeepId2Utils.patchCenters[{{},iPatch}]
		if (k == 2) then
			center[1] = DeepId2Utils.imageCenter[1] - (center[1] - DeepId2Utils.imageCenter[1])
		end
		local patchRadius = (patchSize - 1) / 2
		local topLeft = torch.round(center - patchRadius)
		local bottomRight = topLeft + patchSize - 1
		if (k==1) then
			DeepId2Utils.patchBordes[{{},{iPatch}}] = torch.Tensor({topLeft[2],bottomRight[2],topLeft[1],bottomRight[1]})
		elseif (k == 2) then
			DeepId2Utils.patchBordesFlipped[{{},{iPatch}}] = torch.Tensor({topLeft[2],bottomRight[2],topLeft[1],bottomRight[1]})
		end
	end	
end

function DeepId2Utils.getPatchInner(images, patchBorders, patchSizeTarget, isRgb, isFlipped)
    -- crop patches
    patches = images[{{},{},{patchBorders[1], patchBorders[2]},
        {patchBorders[3], patchBorders[4]}}]
		
    -- resize patches to  patchSizeTarget
    nImages = images:size(1)

	local patchesResized
	if isRgb then
		patchesResized = torch.Tensor(nImages, 3,
			patchSizeTarget[2], patchSizeTarget[1])
	else 
		patchesResized = torch.Tensor(nImages, 1,
			patchSizeTarget[2], patchSizeTarget[1])
	end	
	
    for iImage = 1,nImages do
		if isRgb then
			patchesResized[iImage] = image.scale(patches[iImage], patchSizeTarget[2], patchSizeTarget[1])
		else
			patchesResized[iImage] = image.scale(image.rgb2y(patches[iImage]), patchSizeTarget[2], patchSizeTarget[1])
		end
		if isFlipped then
			patchesResized[iImage] = image.hflip(patchesResized[iImage])
		end	
	end
    return patchesResized
end

function DeepId2Utils.getPatch(images, iPatch, useFlipped)
    -- images : tensor of format nImages x 3 x height x width
    -- iPatch : indicating the requested patch (1-5)

	nImages = images:size(1)
    patchBorders =  DeepId2Utils.patchBordes[{{},iPatch}]
    patchSizeTarget =  DeepId2Utils.patchSizeTarget[iPatch]
    if  DeepId2Utils.verbose then
        print(' DeepId2Utils.getPatch : index=', iPatch)
        print('patch size = ', patchSizeTarget)
    end
	if (type(patchSizeTarget) == 'number') then
      patchSizeTarget = {patchSizeTarget, patchSizeTarget}
    end
	
	patchesResized = DeepId2Utils.getPatchInner(images, patchBorders, patchSizeTarget, DeepId2Utils.isRgb[iPatch], DeepId2Utils.isFlipped[iPatch])
	if DeepId2Utils.isRgb[iPatch] then
		nChannels = 3
	else
		nChannels = 1
	end
		
    local flipeedPatchResized
    local patchesResizedTotal
    -- data augmentation : using flipped patches also
    if useFlipped then
		-- NOTE: we use "not isFlipped" for the mirror patch to get the same horizontal direction
		flipeedPatchResized = DeepId2Utils.getPatchInner(images, DeepId2Utils.patchBordesFlipped[{{},iPatch}], patchSizeTarget, 
			DeepId2Utils.isRgb[iPatch], not DeepId2Utils.isFlipped[iPatch])

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