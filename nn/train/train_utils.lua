function initializeTrainTools()
    -- defining output paths
    state_file_path = paths.concat(opt.save, 'model.net')
    state_file_last_path = paths.concat(opt.save, 'model_last.net')
    train_log_path = paths.concat(opt.save, 'train.log')
    test_log_path = paths.concat(opt.save, 'test.log')

    -- Log results to files
    trainLogger = optim.Logger(train_log_path, true)
    testLogger = optim.Logger(test_log_path, true)
    timestamp = os.date("%Y_%m_%d_%X")
    ----------------------------------------------------------------------
    optimState = {
        learningRate = opt.learningRate,
        weightDecay = opt.weightDecay,
        momentum = opt.momentum,
        learningRateDecay = opt.learningRateDecay
    }
    if (opt.nesterov) then
        optimState.nesterov = true
        optimState.dampening = 0
    end
    print '==> configuring optimizer - SGD'
    optimMethod = optim.sgd

    -- This matrix records the current confusion across classes
    confusion = optim.ConfusionMatrix(classes)
end

-- small wrapper to optim.sgd, with more convenient syntax
function doSgd(x, dfdx, optimState)
    local opfunc = function()
        return 0,dfdx
    end
    optim.sgd(opfunc, x, optimState)
end