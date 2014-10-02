function getOptions()
	cmd = torch.CmdLine()
	cmd:text()
	cmd:text('Deepface training')
	cmd:text()
	cmd:text('Options:')

	-- global:
	cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
	cmd:option('-threads', 2, 'number of threads')
	cmd:option('-visualize', false, 'visualize input data and weights during training')
    cmd:option('-loadState', false, 'load exisiting state : pre-trained net and optimization parameters')

	-- data:
	cmd:option('-size', 'full', 'how many samples do we load: small | full')
    cmd:option('-dataFormat', 't7', 'dataset file type: mat | t7')

	-- training:
	-- TODO: update to defaults values to match deepface training params
	cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
	cmd:option('-plot', false, 'live plot')

		-- optimization parameters (same as Krizhevsky ImageNet)
	cmd:option('-learningRate', 0.01, 'learning rate at t=0')
	cmd:option('-learningRateDecay', 0, 'learning rate decay')
	cmd:option('-batchSize', 128, 'mini-batch size (1 = pure stochastic)')
	cmd:option('-weightDecay', 5e-4, 'weight decay for SGD')
	cmd:option('-momentum', 0.9, 'momentum for SGD')
	cmd:text()
	opt = cmd:parse(arg or {})
	return opt
end
