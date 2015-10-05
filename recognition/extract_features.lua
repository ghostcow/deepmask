----------------------------------------------------------------------
-- Run the network over the given dataset
----------------------------------------------------------------------

package.path = package.path .. ";" .. '../nn_utils/?.lua'
require 'dataset'
require 'mattorch'
require 'paths'
require 'xlua'
require 'nn'
require 'cunn'
require 'cudnn'
require 'stn'

torch.setnumthreads(12)
torch.manualSeed(11)
torch.setdefaulttensortype('torch.FloatTensor')

function getOptions()
    local cmd = torch.CmdLine()

    cmd:text()
    cmd:text('Face feature extractor')
    cmd:text()
    cmd:text('Options:')

    cmd:option('--dataPath', '../results/dataset.t7', 'path to dataset file path (created using dataset.lua)')
    cmd:option('--modelPath', '../results/nn/model_best.net', 'path to nn model')
    cmd:option('--outputPath', '../results/features', 'path to lfw directory or dataset')
    cmd:option('--batch', 128, 'batch size')
    cmd:option('--nn', false, 'path to lfw directory or dataset')
    cmd:option('--cpu', false, 'should use cuda')

    local opt = cmd:parse(arg or {})
    return opt
end

function loadModel(modelPath, opt)
    print '==> loading model'
    local model = torch.load(modelPath)
    if not opts.cpu then
        model:cuda()
    end
    model:evaluate()

    -- remove classifier layers
    local n = 1
    while (torch.type(model.modules[n]) ~= 'nn.Linear') and (n <=  #model.modules) do
        n = n + 1
    end

    if opt.nn then
        n = n + 1
    end

    -- for full_conv model start from i=31
    for i = n,#model.modules do
        model.modules[i] = nil
    end

    print(model)

    return model
end

function getEmbeddingSize(model, dataset)
    local sample, _, _ = dataset:get(1,2)
    print(sample:size())
    local output = model:forward(sample:cuda())

    return output:nElement()/2
end

function extract_features(dataset, model, featuresFilename)
    local features = torch.Tensor(dataset:sizeTest(), getEmbeddingSize(model, dataset))
    local features_labels = torch.Tensor(dataset:sizeTest())

    local t = 1
    local batchSize
    if opts.cpu then
        batchSize = 1
    else
        batchSize = opts.batch
    end

    for inputs, labels in dataset:test(batchSize) do
        -- disp progress
        xlua.progress(t, dataset:sizeTest())

        if not opts.cpu then
            inputs = inputs:cuda()
        end

	    features[{{t,t+inputs:size(1)-1}}]:copy(model:forward(inputs):float())
        features_labels[{{t,t+inputs:size(1)-1}}] = labels

        -- grabage collection after every batch and advance
        t = t + inputs:size(1)
        collectgarbage()
    end

    mattorch.save(featuresFilename, {features = features:double(),
                                     labels = features_labels:double()})
end

function main()
    opts = getOptions()

    print '==> loading test dataset'
    local dataset = torch.dataset.load(nil, opts.dataPath)
    local model = loadModel(opts.modelPath, opts)

    print('==> creating feature directory: ' .. opts.outputPath)
    paths.mkdir(opts.outputPath)

    print '==> extracting features from test set'
    extract_features(dataset, model, opts.outputPath .. '/test_features.mat')
end


main()

