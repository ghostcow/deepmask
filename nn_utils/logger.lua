require 'paths'
require 'optim'
local tds = require 'tds'
tds.hash.__ipairs = tds.hash.__pairs

----------------------------------------------------------------------
-- defining output paths
local logDir = paths.concat(opt.save, 'nn')
paths.mkdir(logDir)

local train_log_path = paths.concat(logDir, 'train.log')
local test_log_path = paths.concat(logDir, 'test.log')
local costs_log_path = paths.concat(logDir, 'costs.log')

-- Log results to files
trainLogger = optim.Logger(train_log_path, true)
testLogger = optim.Logger(test_log_path, true)
costsLogger = nil
bestAccuracy = 0
bestAccCounter = 0

function logCosts(nllTotalErr, kdTotalErr, totalErr)
    if not costsLogger then
        costsLogger = optim.Logger(costs_log_path, true)
    end

    costsLogger:add{['% NLL cost function value'] = nllTotalErr,
                    ['% KD cost function value'] = kdTotalErr,
                    ['% total cost function value'] = totalErr}
end

function logConfusion(confusion, totalMaskError, totalScoreError, saveConfusion)
    -- print confusion matrix  & error funstion values
    confusion:updateValids()
    print("Score Branch Accuracy = ", confusion.totalValid * 100)

    if saveConfusion ~= nil then
        local filename_confusion = paths.concat(logDir, 'confusion_train')
        torch.save(filename_confusion, confusion)
    end

    print('Mask Branch Error = '..tostring(totalMaskError))
    print('Score Branch Error = '..tostring(totalScoreError))
    trainLogger:add{['% total accuracy'] = confusion.totalValid * 100,
                    ['% average accuracy'] = confusion.averageValid * 100,
                    ['% mask loss error'] = totalMaskError,
                    ['% score loss error'] = totalScoreError}
    if opt.plot then
        trainLogger:style{['% total accuracy'] = '-'}
        trainLogger:plot()
    end
end

function trimModel(trainedModel)
    if trainedModel.syncParameters ~= nil then
        trainedModel = trainedModel.modules[1]
    end

    for i=1,#trainedModel.modules do
        local layer = trainedModel:get(i)
        if layer.output ~= nil then
            layer.output = layer.output.new()
        end
        if layer.gradInput ~= nil then
            layer.gradInput = layer.gradInput.new()
        end
        -- for cudnn layer we need to reset the oDesc and iDesc
        if layer.oDesc ~= nil or layer.iDesc ~= nil then
            layer.oDesc = nil
            layer.iDesc = nil
        end

        collectgarbage()
    end

    return trainedModel:float():clone()
end

function logNetwork(trainedModel, modelName, originalType)
    if modelName == nil then
        modelName = 'model'
    end

    local state_file_path = paths.concat(logDir, modelName .. '.net')
    local state_file_last_path = paths.concat(logDir,  modelName .. '_last.net')

    -- save/log current net
    print('==> saving model to '..state_file_path)
    os.rename(state_file_path, state_file_last_path)

    -- transfer model to RAM, clone, save, return to CudaTensor (or previous type)
    local fmodel = trimModel(trainedModel)
    torch.save(state_file_path, fmodel)
    trainedModel:type(originalType)
end

function logOptimState(optimState, modelName)
    if modelName == nil then
        modelName = 'model'
    end

    local optim_state_file_path = paths.concat(logDir, modelName .. '.optim_state')
    -- save/log current state
    print('==> saving optim state to '.. optim_state_file_path)
    torch.save(optim_state_file_path, optimState)
end

function logTest(confusion, modelName, model)
    -- log and plot accuracy
    print("accuracy = ", confusion.totalValid * 100)
    testLogger:add{['% total accuracy']   = confusion.totalValid * 100,
                   ['% average accuracy'] = confusion.averageValid * 100 }

    if opt.plot then
        testLogger:style{['% total accuracy'] = '-'}
        testLogger:plot()
    end

    -- save best model
    if bestAccuracy < confusion.averageValid then
        bestAccuracy = confusion.averageValid
        bestAccCounter = 0

        if modelName == nil then
            modelName = 'model'
        end

        local state_file_path = paths.concat(logDir, modelName .. '.net')
        local optim_state_file_path = paths.concat(logDir, modelName .. '.optim_state')

        local best_state_file_path = paths.concat(logDir, modelName .. '_best.net')
        local best_optim_state_file_path = paths.concat(logDir, modelName .. '_best.optim_state')

        if model~=nil and not paths.filep(state_file_path) then
            local fmodel = trimModel(model)
            torch.save(state_file_path, fmodel)
            torch.save(optim_state_file_path, nil)
        end

        if paths.filep(best_state_file_path) then
            os.remove(best_state_file_path)
            os.remove(best_optim_state_file_path)
        end

        os.execute('cp %s %s' % {state_file_path, best_state_file_path})
        os.execute('cp %s %s' % {optim_state_file_path, best_optim_state_file_path})
    else
        bestAccCounter = bestAccCounter + 1
    end
end