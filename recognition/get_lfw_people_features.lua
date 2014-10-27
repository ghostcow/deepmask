package.path = package.path .. ";../nn/?.lua"
require 'lfw_utils'
require 'get_face_features'
require 'options'
require 'nn'
require 'cunn'
require 'ccn2'
require 'mattorch'

featureSize = 4096
---Load model -----------------------------------------------------------------------------------------------
opt = getOptions()
local state_file_path = paths.concat('../results/', opt.save, 'model.net')
model = torch.load(state_file_path)
print(model)
featureLayerIndex = #(model.modules) - 3 -- last 3 layers : dropout, fully conected, log

---Load LFW data and extract face feature -------------------------------------------------------------------
imagePaths = LfwUtils.loadPeople()
print(#imagePaths)
faceFeatures = getFaceFeatures(imagePaths, model, featureLayerIndex, 2)
faceFeaturesMat = torch.Tensor(#faceFeatures, featureSize)
for iFeature,feature in pairs(faceFeatures) do
    faceFeaturesMat[{iFeature}] = feature
end
mattorch.save('lfw_people.mat', faceFeaturesMat:double())