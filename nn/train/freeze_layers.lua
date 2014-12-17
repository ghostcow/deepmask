function freezeLayers(freezeLayers)

    modules = model.modules
    shouldFreezeLayers = torch.LongTensor(#modules):fill(0)
    for layerIdStr in string.gmatch(freezeLayers, '([^,]+)') do
        layerId = tonumber(layerIdStr)
        shouldFreezeLayers[layerId] = 1
    end
    learningRates = torch.FloatTensor(gradParameters:size(1)):fill(1)

    x_index = 1
    for iLayer = 1,#modules do
        numParamsModule = 0
        if (modules[iLayer].weight) then
            numParamsModule = numParamsModule + modules[iLayer].weight:size(1)*modules[iLayer].weight:size(2)
        end
        if (modules[iLayer].bias) then
            numParamsModule = numParamsModule + modules[iLayer].bias:size(1)
        end
        if (shouldFreezeLayers[iLayer]==1) then
            print('freeze params of layer - ', iLayer)
            learningRates[{{x_index,(x_index + numParamsModule - 1)}}] = 0
        end
        print(iLayer, ' = ', x_index, ' : ', x_index + numParamsModule)
        x_index = x_index + numParamsModule
    end
    optimState.learningRates = learningRates
end