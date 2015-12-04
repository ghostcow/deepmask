require 'paths'

function getOptions()
	local cmd = torch.CmdLine()
	cmd:text()
	cmd:text('Deepmask training')
	cmd:text()
	cmd:text('Options:')

	-- global:
	cmd:option('--seed', 1, 'fixed input seed for repeatable experiments')
	cmd:option('--visualize', false, 'visualize input data and weights during training')
    cmd:option('--retrain', 'none', 'load exisiting state : pre-trained net')
    cmd:option('--loadState', 'none', 'load optim state')
    cmd:option('--netType', 'deepmask', 'name of the model (network) to use')
    cmd:option('--gpu', -1, 'Which GPU to use')
    cmd:option('--debug', false, 'work on small dataset')

	-- data:
    cmd:option('--splitName', 'train2014', 'split name')
	cmd:option('--dataPath', '', 'path to dataset files - *.tds.t7 files')
    cmd:option('--imageDirPath', '/Users/adamp/Research/Data/CASIA/CASIA-WebFace', 'path to dataset directory')
    cmd:option('--nWorkers', 4, 'number of threads used for data loading')

	-- training:
    cmd:option('--negativeRatio', 0.5, 'ratio between positive and negative samples in score branch')
    cmd:option('--testOnly', false, 'should test')
	cmd:option('--save', '../results', 'subdirectory to save/log experiments in')
	cmd:option('--plot', false, 'live plot')
    cmd:option('--epochs', 1000, 'number of epochs to train')
    cmd:option('--parallel', false, 'train in parallel mode')

	-- optimization parameters (same as Krizhevsky ImageNet)
    cmd:option('--learningRate', 0.01, 'learning rate at t=0')
    cmd:option('--batchSize', 128, 'mini-batch size (1 = pure stochastic)')
    cmd:option('--epochSize', 2300, 'number of mini-batches per epoch')
    cmd:option('--weightDecay', 5e-4, 'weight decay for fully connected layer')
    cmd:option('--momentum', 0.9, 'momentum for SGD')

    cmd:text()
    
    local opt = cmd:parse(arg or {})

    paths.mkdir(opt.save)
    -- append random number to log name
    local r = torch.random() % 10000
    cmd:log(paths.concat(opt.save, 'doall.'..tostring(r)..'.log'), opt)
    return opt
end

function string:split(sep)
    local sep, fields = sep or ":", {}
    local pattern = string.format("([^%s]+)", sep)
    self:gsub(pattern, function(c) fields[#fields+1] = c end)
    return fields
end

function parseImageSize(imageSize)
    imageSize = imageSize:split('x')
    for k, v in ipairs(imageSize) do
       imageSize[k] = tonumber(v)
    end

    return imageSize
end
