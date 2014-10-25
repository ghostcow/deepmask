--- using pre-trained network to produce face features and test performance over the LFW benchmark
-- NOTE : this script uses nn/options.lua to parse command line arguments, but actually 2 arguments are relevant :
--      opt.save : path to results directory, where the network file will be loaded from
--      opt.modelName : the model name for the used network. used to determine the index of the face feature layer

package.path = package.path .. ";../nn/?.lua"
require 'lfw_utils'
require 'get_face_features'
require 'chi_squared'
require 'options'
require 'nn'
require 'cunn'
require 'ccn2'
require 'mattorch'

featureSize = 4096
faceFeaturesPath = '/media/data/datasets/LFW/view2/pairs_features.t7'
faceFeaturesMatPath = '/media/data/datasets/LFW/view2/pairs_features.mat'

---Load model -----------------------------------------------------------------------------------------------
opt = getOptions()
local state_file_path = paths.concat(opt.save, 'model.net')
model = torch.load(state_file_path)
    -- TODO : load normFactors together with the model
if (opt.modelName == 'model') then
    featureLayerIndex = 16
elseif (opt.modelName =='model2') then
    featureLayerIndex = 12
elseif (opt.modelName =='model_N4') then
    featureLayerIndex = 12
end

---Load LFW data and extract face feature -------------------------------------------------------------------
pairsData = LfwUtils.loadPairs()
imagePaths = LfwUtils.getImagePaths(pairsData)
    -- faceFeatures : key = image path, value = face feature
if os.rename(faceFeaturesPath, faceFeaturesPath) then
    faceFeatures = torch.load(faceFeaturesPath)
else
    faceFeatures = getFaceFeatures(imagePaths, model, featureLayerIndex)
    torch.save(faceFeaturesPath, faceFeatures)

    --- save into mat file also
    --- currently this saving fails for some reason
    faceFeaturesMat = {}
    for path,feature in pairs(faceFeatures) do
        -- save only image name to get valid matlab variable
        local pathCropeed = path:sub(#(LfwUtils.mainDir) + 2)
        local k = pathCropeed:find('/')
        pathCropeed = pathCropeed:sub(k+1, -5)
        print(pathCropeed)

        -- faceFeaturesMat[path] = feature:double()
        faceFeaturesMat[pathCropeed] = feature:double()
        -- table.insert(faceFeaturesMat, feature:double())
    end
    -- mattorch.save(faceFeaturesMatPath, faceFeaturesMat)
end

---SVM train & test------------------------------------------------------------------------------------------
trainSize = (LfwUtils.numFolds - 1)*2*LfwUtils.numPairsPerFold
trainData = torch.FloatTensor(trainSize, 2, featureSize)
trainLabels = torch.LongTensor(trainSize) -- +1 for match pair, -1 for mismatch
accuracies = torch.FloatTensor(LfwUtils.numFolds)
for iTestFold = 1,LfwUtils.numFolds do
    print('testing over fold no.',iTestFold)
    --- collect data for training
    foldIndex = 1
    for iFold = 1,LfwUtils.numFolds do
        if (iFold ~= iTestFold) then
            print('collecting', iFold)
            fold = pairsData[iFold]
            foldData, foldLabels = LfwUtils.getFoldPairsFeatures(fold, faceFeatures)

            trainData[{{foldIndex, foldIndex+2*LfwUtils.numPairsPerFold-1}}] = foldData
            trainLabels[{{foldIndex, foldIndex+2*LfwUtils.numPairsPerFold-1}}] = foldLabels
            foldIndex = foldIndex + 2*LfwUtils.numPairsPerFold
        end
    end
    testFold = pairsData[iTestFold]
    testData, testLabels = LfwUtils.getFoldPairsFeatures(testFold, faceFeatures)

    if (iTestFold == 1) then
        --- save data into mat files
        mattorch.save('trainData.mat', trainData:double())
        mattorch.save('trainLabels.mat', trainLabels:double())
        mattorch.save('testData.mat', testData:double())
        mattorch.save('testLabels.mat', testLabels:double())
    end
    --- TODO : normalize both trainData & testData, based on trainData values

    --- svm training
    print('training svm classifier...')
    classifier = trainChiSquared(trainData, trainLabels)
    print(classifier)

    --- test the classifier
    print('testing...')
    l,acc,d = predictChiSquared(classifier, testData, testLabels)
    accuracies[iTestFold] = acc[1]
end
print('average accuracy = ', accuracies:mean())