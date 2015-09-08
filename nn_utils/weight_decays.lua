local function countLayerparams(layer)
    local numParamsModule = 0

    if (layer.weight) then
        numParamsModule = numParamsModule + layer.weight:nElement()
    end

    if (layer.bias) then
        numParamsModule = numParamsModule + layer.bias:nElement()
    end

    return numParamsModule
end

function getWeightDecays(weightDecay, parameters, gradParameters)
    local modules = model.modules
    local layerParamIndex = 1
    local weightDecays = torch.Tensor():typeAs(parameters):resizeAs(gradParameters):zero()

    for iLayer = 1,#modules do
        local numParamsModule = 0
        local layerType = torch.type(modules[iLayer])

        if layerType == 'nn.Sequential' then
            for i = 1,#modules[iLayer].modules do
                numParamsModule = numParamsModule + countLayerparams(modules[iLayer].modules[i])
            end
        else
            numParamsModule = countLayerparams(modules[iLayer])
        end

        if layerType:find("Linear") then
            print('Weight decay params for layer ' .. iLayer .. ' is ' .. weightDecay)
            weightDecays[{{layerParamIndex,(layerParamIndex + numParamsModule - 1)}}] = weightDecay;
        end
        print(iLayer .. ' = ' .. layerParamIndex .. ' : ' .. layerParamIndex + numParamsModule)
        layerParamIndex = layerParamIndex + numParamsModule
    end

    return weightDecays
end