require 'paths'

function getOptions()
	local cmd = torch.CmdLine()
	cmd:text()
	cmd:text('Face recongnition training')
	cmd:text()
	cmd:text('Options:')

	-- global:
	cmd:option('--seed', 1, 'fixed input seed for repeatable experiments')
	cmd:option('--threads', 2, 'number of threads')
	cmd:option('--visualize', false, 'visualize input data and weights during training')
    cmd:option('--retrain', 'none', 'load exisiting state : pre-trained net')
    cmd:option('--loadState', 'none', 'load optim state')
    cmd:option('--netType', 'scratch', 'name of the model (network) to use')
    cmd:option('--gpu', -1, 'Which GPU to use')

	-- data:
	cmd:option('--dataPath', '', 'path to dataset file - t7 file')
    cmd:option('--imageDirPath', '/Users/adamp/Research/Data/CASIA/CASIA-WebFace', 'path to dataset directory')
    cmd:option('--imageSize', '1x100x100', 'size of sample images')
    cmd:option('--split', 30, 'precentage of data used for test')
    cmd:option('--nDonkeys', 4, 'number of threads used for data loading')

	-- training:
    cmd:option('--testOnly', false, 'should test')
	cmd:option('--save', '../results', 'subdirectory to save/log experiments in')
	cmd:option('--plot', false, 'live plot')
    cmd:option('--epochs', 1000, 'number of epochs to train')

	-- optimization parameters (same as Krizhevsky ImageNet)
    cmd:option('--learningRate', 0.01, 'learning rate at t=0')
    cmd:option('--batchSize', 128, 'mini-batch size (1 = pure stochastic)')
    cmd:option('--weightDecay', 5e-4, 'weight decay for fully connected layer')
    cmd:option('--momentum', 0.9, 'momentum for SGD')

    -- data augmentation
    cmd:option('--blurSize', -1, 'gaussian kernel size')
    cmd:option('--blurSigma', 10, 'gaussiam sigma param')

    cmd:text()
    
    local opt = cmd:parse(arg or {})
    opt.imageSize = parseImageSize(opt.imageSize)
    print(opt.imageSize)

    paths.mkdir(opt.save)
    cmd:log(paths.concat(opt.save, 'doall.log'), opt)
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