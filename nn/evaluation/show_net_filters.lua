require 'nn'
require 'cunn'
require 'ccn2'
require 'optim'
require 'gfx.js'

layersIds = {C1=2,C3=5,L4=7,L5=9,L6=11,F7=15,F8=18}
filtersSize = {11, 9, 9, 7, 5 }
numMaps = {32, 16, 16, 16, 16 }
inputDim = {3, 32}

--
print '--- Loading model ---'
model_data = torch.load('../results_small_model2/model.net')
model = model_data.model:float()

-- C1 : 32x11x11x3, weights format : (11*11*3)x32
-- C3 : 16x9x9x32, weights format : (9*9*32)x16
layersIndices = {2, 5 }
layerNames = {'C1','C2'}
for iLayer = 1,2 do
    weights = model:get(layersIndices[iLayer]).weight
    nFilters = numMaps[iLayer]

    for iInputDim = 1,inputDim[iLayer] do
        filters = torch.Tensor(nFilters, filtersSize[1], filtersSize[1])
        iStart = 1 + (iInputDim-1)*filtersSize[1]*filtersSize[1]
        for iFilter = 1,nFilters do
            local currFilterFlat = weights[{{},{iFilter}}]
            currFilterCurrChannel = currFilterFlat[{{iStart,(iStart+filtersSize[1]*filtersSize[1]-1)}}]
            currFilterCurrChannel = currFilterCurrChannel:resize(filtersSize[1], filtersSize[1])
            filters[{{iFilter}}] = currFilterCurrChannel
        end
        gfx.image(filters, {zoom=10, legend=layerNames[iLayer]..' - channel '..tostring(iInputDim)})
    end
end


