package.path = package.path .. ";../nn/?.lua;../nn/deepId/?.lua"
require 'image'
require 'io'
require 'mattorch'
require 'math'
require 'nn'
require 'cunn'
require 'ccn2'
require 'options'

-- run this script like this :
-- th get_deepid_features.lua <output path> --save <results dir name> --dataPath <input mat file path>

useFlippedPatches = true
--- first input arguments is mandatory for this script
print(arg)
outputPath = arg[1] -- output mat file path
arg[1] = nil
for iArg = 2,#arg do
    arg[iArg-1] = arg[iArg]
    arg[iArg] = nil
end
opt = getOptions()
opt.save = paths.concat('../results_deepid/', opt.save)

print('Loading input data :', opt.dataPath)
dataSet = mattorch.load(opt.dataPath)
dataSetSize = dataSet.data:size()[4]

dataSet = {
    -- the original matlab format is nImages x 3 x height x width
    -- (where height=width=152)
    -- but it's loaded into torch like this : width x height x 3 x nImages
    data = dataSet.data:transpose(1,4):transpose(2,3),
    labels = dataSet.labels[1],
    size = function() return dataSetSize end
}
print(dataSet)
if useFlippedPatches then
    -- replicate labels
    dataSet.labels = torch.cat(dataSet.labels, dataSet.labels)
end

require 'deep_id_utils'
print('networks location : ', opt.save)
print 'applying networks over the data'
--- iterate the different networks
for modelDirName in io.popen('ls -a "'..opt.save..'"'):lines() do
    print(modelDirName)
    if (#modelDirName > #'patch' and (modelDirName:sub(1,#'patch') == 'patch')) then
    print('valid directory')
    modelDirPath = paths.concat(opt.save, modelDirName)
    model = torch.load(paths.concat(modelDirPath, 'model.net'))
    model:evaluate() -- turn off all dropouts
    featureLayerIndex = #(model.modules) - 3 -- last 3 layers : dropout, fully conected, log

    opt.patchIndex = tonumber(modelDirName:sub(#'patch'+1))
    print(opt.patchIndex)
    dataSetPatch = DeepIdUtils.getPatch(dataSet.data, opt.patchIndex, useFlippedPatches)
    dataSetPatchSize = dataSetPatch:size(1)
    imageDim = {dataSetPatch:size(3), dataSetPatch:size(4)}

    features = torch.Tensor() -- will be resized later
    -- iterate over batches
    for t = 1,dataSetPatchSize,opt.batchSize do
        -- disp progress
        xlua.progress(t, dataSetPatchSize)
        inputs = torch.Tensor(opt.batchSize, 3, imageDim[1], imageDim[2])
        for i = t,(t+opt.batchSize-1) do
            -- apply mod operator, in order to complete last batch with first samples
            inputs[{i-t+1}] = dataSetPatch[1 + (i-1)%dataSetPatchSize]
        end
        inputs = inputs:cuda()
        outputs = model:forward(inputs)
        featuresBatch = model:get(featureLayerIndex).output:float()
        if (features:dim() == 0) then
            featureDim = featuresBatch:size(2)
            features:resize(dataSetPatchSize, featureDim)
        end
        for i = t,math.min(dataSetPatchSize, (t+opt.batchSize-1)) do
            if (dataSet.labels[i] == 0) then
                -- invalid image, fill feature with zeros
                print('invalid image - ', i)
                featuresBatch[i-t+1] = torch.Tensor(1, featureDim):fill(0)
            end
            features[i] = featuresBatch[i-t+1]
        end
    end
    print(features:size())
    if useFlippedPatches then
        -- concat feature of each patch and its flipped version
        features = torch.cat(features[{{1,dataSetPatchSize/2},{}}], features[{{dataSetPatchSize/2+1,dataSetPatchSize},{}}], 2);
    end
    print(features:size())

    mattorch.save(outputPath..'_'..modelDirName..'.mat', features)
    end
end
