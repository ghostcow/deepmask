-- loading model trained on cfw_small and copy all layer weights (except the last one) into model fit to full cfw

sourceModelPath = '../results/results_cfw_small2_model2/model.net'
targetModelPath = '../results/cfw_clean_model2_init_cfw_small2/model.net'

print '==> initializing model with 556 persons'
nLabels = 556
dofile 'model2.lua'
print '==> changing model to CUDA'
model:add(nn.LogSoftMax())
model:cuda()

print '==> loading pre-trained model'
-- layersIds = {C1=2,C3=5,L5=7,F7=11,F8=14}
model_trained = torch.load(sourceModelPath)

print '==> copying relevant layers (all except F8)'
-- copy layers weights
layerIds = {2,5,7,11} -- {C1=2,C3=5,L5=7,F7=11,F8=14}
for iLayer = 1,4 do
    weight = model_trained:get(layerIds[iLayer]).weight
    bias = model_trained:get(layerIds[iLayer]).bias

    model:get(layerIds[iLayer]).weight:copy(weight)
    model:get(layerIds[iLayer]).bias:copy(bias)
end

print '==> saving new model'
torch.save(targetModelPath, model)
