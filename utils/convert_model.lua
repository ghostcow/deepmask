-- loading model trained on one dataset and copy all layer weights (instead of the last fully connected layer)
-- into a new model with different class labels
-- allowed conversions : model --> model, model2 -->model2, ...

modelName = 'model_N4'
sourceModelPath = '../results/cfw_small2_model_N4/model.net'
targetModelPath = '../results/cfw_clean_model_N4_init_cfw_small2/model.net'
nLabels = 588

-- indices of the layers which will be copied
layerIds = {2,5,7,11} -- {C1=2,C3=5,L5=7,F7=11,F8=14}

print '==> initializing model'
dofile(modelName..'.lua')
print '==> changing model to CUDA'
model:add(nn.LogSoftMax())
model:cuda()

print '==> loading pre-trained model'
-- layersIds = {C1=2,C3=5,L5=7,F7=11,F8=14}
model_trained = torch.load(sourceModelPath)

print '==> copying relevant layers (all except F8)'
for iLayer = 1,#layerIds do
    weight = model_trained:get(layerIds[iLayer]).weight
    bias = model_trained:get(layerIds[iLayer]).bias

    model:get(layerIds[iLayer]).weight:copy(weight)
    model:get(layerIds[iLayer]).bias:copy(bias)
end

print '==> saving new model'
torch.save(targetModelPath, model)
