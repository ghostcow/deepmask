function getOptions()
	cmd = torch.CmdLine()
	cmd:text()
	cmd:text('Deepface training')
	cmd:text()
	cmd:text('Options:')

	-- global:
	cmd:option('--seed', 1, 'fixed input seed for repeatable experiments')
	cmd:option('--threads', 2, 'number of threads')
	cmd:option('--visualize', false, 'visualize input data and weights during training')
    cmd:option('--loadState', false, 'load exisiting state : pre-trained net and optimization parameters')
    cmd:option('--modelName', 'model', 'name of the model (network) to use - model/model2/model_clean_aligned_faces')
    cmd:option('--debugOnly', false, 'if true, no trainig-testing is done and everthing is just loaded')

    -- deepid parameters :
    cmd:option('--patchIndex', 1, 'relevant for deepId only - index of the patch for training')
    cmd:option('--deepIdMode', 1, 'mode which indicates how the patches are generated (1=gets deepface images(152x152) and produce 5 patches, 2=gets deepid images(140x115) and produce 15 patches (3 scales)')

	-- data:
	cmd:option('--dataPath', '../data_files/CFW_flat/cfw_flat.t7', 'path to dataset file (mat or t7)')
    cmd:option('--useDatasetChunks', false, 'read datasets in chunks')
    cmd:option('--numPassesPerChunk', 1, 'how many passes to do over one chunk before loading the next one')

	-- training:
	cmd:option('--save', '../results', 'subdirectory to save/log experiments in')
	cmd:option('--plot', false, 'live plot')
    cmd:option('--balanceClasses', false, 'whether to use balance cost for classes in the NLL criterion')
    cmd:option('--freezeLayers', '', 'layers indices (seperated by comma) whose parameters will not be updated during training')
    cmd:option('--trainOnly', false, 'whether to use all data for training (with no test phase at all)')
    cmd:option('--loss', 'identification', 'which loss function to use (identification/combined)')

    -- verification (contrastive) cost
    cmd:option('--lambda', 0.05, 'verification cost weight')
    cmd:option('--margin', 100, 'initial margin in the verification cost')

	-- optimization parameters (same as Krizhevsky ImageNet)
	cmd:option('--learningRate', 0.01, 'learning rate at t=0')
	cmd:option('--learningRateDecay', 0, 'learning rate decay')
	cmd:option('--batchSize', 128, 'mini-batch size (1 = pure stochastic)')
	cmd:option('--weightDecay', 5e-4, 'weight decay for SGD')
	cmd:option('--momentum', 0.9, 'momentum for SGD')
    cmd:option('--nesterov', false, 'whether to use nesterov momentum')

    cmd:text()
	opt = cmd:parse(arg or {})

    print('opt : ')
    print(opt)
	return opt
end
