-- allowed conversions : model_N4 --> model_N5, model_N5 --> model

modelName = 'model_N5'
sourceModelPath = '../results/cfw_small2_model_N4/model.net'
targetModelPath = '../results/cfw_small2_model_N5/model.net'
nLabels = 32

inputDim = 63 -- output of C3
filterSize = 9
numMaps = 16
sourceStride = 4
targetStride = 1

-- {C1=2,C3=5,L4=7,F7=11,F8=14}
freezeLayersIndices = {2,5}
partlyCopiedLayerIndex = 7

-----------------------------------------------------------------------------------------------------------
print '==> initializing model'
dofile(modelName..'.lua')
print '==> changing model to CUDA'
model:add(nn.LogSoftMax())
model:cuda()
print '==> loading pre-trained model'
model_trained = torch.load(sourceModelPath)

--------------------------------- copying specified layers
print '==> copying first layers weights'
for iLayer = 1,#freezeLayersIndices do
    weight = model_trained:get(layerIds[iLayer]).weight
    bias = model_trained:get(layerIds[iLayer]).bias

    model:get(layerIds[iLayer]).weight:copy(weight)
    model:get(layerIds[iLayer]).bias:copy(bias)
end

--------------------------------- copying & duplicating locally connected layer
function round(num)
    if num >= 0 then return math.floor(num+.5)
    else return math.ceil(num-.5) end
end

outputSizeTarget = math.ceil((inputDim - filterSize)/targetStride) + 1
outputSizeSource = math.ceil((inputDim - filterSize)/sourceStride) + 1
totalFilterLength = filterSize * filterSize * numMaps
stridesFactor = sourceStride / targetStride

weightSource = model_trained:get(partlyCopiedLayerIndex).weight
biasSource = model_trained:get(partlyCopiedLayerIndex).bias
weightsTarget = model:get(partlyCopiedLayerIndex).weight
biasTarget = model:get(partlyCopiedLayerIndex).bias

-- assuming the filters are organized in column-stack manner
for iCol = 1,outputSizeTarget do
    for iRow = 1,outputSizeTarget do
        iRowSource = 1 + round((iRow - 1) / stridesFactor)
        iColSource = 1 + round((iCol - 1) / stridesFactor)

        startIndexTarget = (iCol - 1) * outputSizeTarget * totalFilterLength +
                (iRow - 1) * totalFilterLength + 1
        startIndexSource = (iColSource - 1) * outputSizeSource * totalFilterLength +
                (iRowSource - 1) * totalFilterLength + 1

        weightsTarget[{{startIndexTarget, startIndexTarget + totalFilterLength - 1}, {}}] =
            weightSource[{{startIndexSource, startIndexSource + totalFilterLength - 1}, {}}]

        -- TODO : add bias copying
    end
end

print(bias:size())
print(totalFilterLength)
print('target')
print(outputSizeTarget)
print(string.format('[%d:%d]', startIndexTarget, startIndexTarget + totalFilterLength - 1))
print('source')
print(outputSizeSource)
print(string.format('[%d:%d]', startIndexSource, startIndexSource + totalFilterLength - 1))

--------------------------------- saving new model
print '==> saving new model'
torch.save(targetModelPath, model)