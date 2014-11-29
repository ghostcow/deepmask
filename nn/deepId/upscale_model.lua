-- loading model trained on specific scale and resizing all filters in the convolutional layers
-- to get new model aimed to deal with the same kinds of pictures in bigger scale

package.path = package.path .. ";../?.lua"
require 'image'
require 'nn'
require 'cunn'
require 'ccn2'
require 'options'
if not opt then
    print '==> processing options'
    opt = getOptions()
end

--------- Constants --------------------------------------------------
sourceModelPath = '../../results_deepid/CFW_PubFig_SUFR_deepID.3.64_15patches/patch6/model.net'
targetModelPath = '../../results_deepid/CFW_PubFig_SUFR_deepID.3.64_15patches/patch11/model.net'
opt.modelName = 'deepID.3.64'
opt.deepIdMode = 2
opt.patchIndex = 11

lcInputSizeSource = 8
lcFilterSizeSource = 5
lcInputSizeTarget = 10
lcFilterSizeTarget = 7
----------------------------------------------------------------------


print '==> loading pre-trained model'
model_trained = torch.load(sourceModelPath)
print('model_trained = ', model_trained)

nModules = #model_trained.modules
nLabels = model_trained:get(nModules - 1).weight:size(1)
print('num labels = ', nLabels)
dofile('../model_deepID.lua')
model:add(nn.LogSoftMax())
model:cuda()
print('model = ', model)

inputMaps = 3
iStartSource = 1
iStartTarget = 1

for iModule = 1,nModules do
    layerType = torch.type(model_trained.modules[iModule]);
    if ((layerType == 'ccn2.SpatialConvolution') or (layerType == 'ccn2.SpatialConvolutionLocal') or
            (layerType == 'nn.Linear')) then

        weights = model_trained:get(iModule).weight
        bias = model_trained:get(iModule).bias

        print(iModule, layerType)
        weightsTarget = model:get(iModule).weight
        biasTarget = model:get(iModule).bias

        print('weights : ', weights:size(), 'bias : ', bias:size())
        print('weightsTarget : ', weightsTarget:size(), 'biasTarget : ', biasTarget:size())

        if (layerType == 'nn.Linear') then
            weightsTarget:copy(weights)
            biasTarget:copy(bias)
        elseif (layerType == 'ccn2.SpatialConvolution') then
            numMaps = weights:size(2)
            filterSize = math.sqrt(weights:size(1) / inputMaps)
            filterSizeTarget = math.sqrt(weightsTarget:size(1) / inputMaps)

            for iMap = 1,numMaps do
                currMapFilter = weights[{{},{iMap}}]
                currMapFilterTarget = weightsTarget[{{},{iMap}}]
                print('Map ', iMap)
                for iInputDim = 1,inputMaps do
                    iStart = 1 + (iInputDim-1)*filterSize*filterSize
                    currFilterCurrChannel = currMapFilter[{{iStart,(iStart+filterSize*filterSize-1)}}]
                    currFilterCurrChannel = currFilterCurrChannel:resize(filterSize, filterSize)
                    currFilterCurrChannel = currFilterCurrChannel:float()

                    -- creating filter for the new model
                    filterTarget = torch.Tensor(filterSizeTarget, filterSizeTarget):fill(0)
                    filterTarget[{{2,filterSizeTarget-1},{2,filterSizeTarget-1}}] = currFilterCurrChannel
                    filterTargetFlat = filterTarget:resize(filterSizeTarget*filterSizeTarget)
                    iStartTarget = 1 + (iInputDim-1)*filterSizeTarget*filterSizeTarget
                    currMapFilterTarget[{{iStartTarget,(iStartTarget+filterSizeTarget*filterSizeTarget-1)}}] =
                        filterTargetFlat:cuda()
                    if (iMap == numMaps) and (iInputDim == inputMaps) then
                        print(filterTarget:resize(filterSizeTarget,filterSizeTarget))
                        print(iStart,(iStart+filterSize*filterSize-1),
                            iStartTarget,iStartTarget+filterSizeTarget*filterSizeTarget-1)
                    end
                end
                biasTarget[iMap] = bias[iMap]
            end
        else
            numMaps = weights:size(2)
            numFilters = weights:size(1) / (inputMaps*lcFilterSizeSource*lcFilterSizeSource)
            numFiltersTarget = weightsTarget:size(1) / (inputMaps*lcFilterSizeTarget*lcFilterSizeTarget)
            for iMap = 1,numMaps do
                currMapFilter = weights[{{},{iMap}}]
                currMapFilterTarget = weightsTarget[{{},{iMap}}]
                -- print('Map ', iMap)
                for iInputDim = 1,inputMaps do
                    for iFilter = 1,numFilters do
                        iStart = 1 + (iInputDim-1)*numFilters*lcFilterSizeSource*lcFilterSizeSource
                                + (iFilter-1)*lcFilterSizeSource*lcFilterSizeSource
                        currFilterCurrChannel = currMapFilter[{{iStart,(iStart+lcFilterSizeSource*lcFilterSizeSource-1)}}]
                        currFilterCurrChannel = currFilterCurrChannel:resize(lcFilterSizeSource, lcFilterSizeSource)
                        currFilterCurrChannel = currFilterCurrChannel:float()

                        -- creating filter for the new model
                        filterTarget = torch.Tensor(lcFilterSizeTarget, lcFilterSizeTarget):fill(0)
                        filterTarget[{{2,lcFilterSizeTarget-1},{2,lcFilterSizeTarget-1}}] = currFilterCurrChannel
                        filterTargetFlat = filterTarget:resize(lcFilterSizeTarget*lcFilterSizeTarget)
                        iStartTarget = 1 + (iInputDim-1)*numFilters*lcFilterSizeTarget*lcFilterSizeTarget
                                + (iFilter-1)*lcFilterSizeTarget*lcFilterSizeTarget
                        currMapFilterTarget[{{iStartTarget,(iStartTarget+lcFilterSizeTarget*lcFilterSizeTarget-1)}}] =
                            filterTargetFlat:cuda()
                        if (iMap == numMaps) and (iInputDim == inputMaps) and (iFilter == numFilters) then
                            print(filterTarget:resize(lcFilterSizeTarget, lcFilterSizeTarget))
                            print(iStart,(iStart+lcFilterSizeSource*lcFilterSizeSource-1),iStartTarget,
                                iStartTarget+lcFilterSizeTarget*lcFilterSizeTarget-1)
                        end
                        biasTarget[(iMap-1)*numFilters + iFilter] = bias[iMap]
                    end
                end
            end
        end
        inputMaps = numMaps
    end
end
print '==> saving new model'
-- torch.save(targetModelPath, model)