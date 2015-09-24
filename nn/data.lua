require 'options'
require 'paths'
require 'dataset'
require 'memory_dataset'
local Threads = require 'threads'

----------------------------------------------------------------------
-- use -visualize to show network
-- parse command line arguments

if not opt then
    print '==> processing options'
    opt = getOptions()
end

----------------------------------------------------------------------
-- used opt options opts.dataPath opts.trainOnly

if opt.dataPath == '' then
    -- create new dataset file
    dataset = torch.dataset{paths={opt.imageDirPath}, sampleSize=opt.imageSize, split=(100 - opt.split)}
    dataset:save(paths.concat(opt.save,'dataset.t7'))
else
    dataset = torch.load(opt.dataPath)
    if  torch.type(dataset) == 'torch.dataset' then
        dataset.sampleHookTrain = dataset.defaultSampleHook
        dataset.sampleHookTest = dataset.defaultSampleHook
    end
end

----------------------------------------------------------------------
-- check if we are in debug
if opt.debug then
    dataset.sizeTrain = function() return 2048 end
    dataset.testIndices = dataset.testIndices[{{1,2048}}]
end

-------------------------------------------------------------------------------
-- This script contains the logic to create K threads for parallel data-loading.
-- For the data-loading details, look at donkey.lua

do -- start K datathreads (donkeys)
    if opt.nDonkeys > 0 then
        -- make an upvalue to serialize over to donkey threads
        local options = opt
        local localDataset = dataset

        localDataset.sampleHookTrain = nil
        localDataset.sampleHookTest = nil

        donkeys = Threads(opt.nDonkeys,
            function()
                package.path = package.path .. ";" .. '../nn_utils/?.lua'
                gsdl = require 'sdl2'
                require 'torch'
                require 'dataset'
                require 'math'
                require 'blur'
            end,

            function(idx)
                -- pass to all donkeys via upvalue
                opt = options
                dataset = localDataset
                tid = idx
                tepoch = 0

                dataset.sampleHookTrain = dataset.defaultSampleHook
                dataset.sampleHookTest = dataset.defaultSampleHook

                -- init thread seed
                local seed = opt.seed + idx
                torch.manualSeed(seed)
                print(string.format('Starting donkey with id: %d seed: %d', tid, seed))
            end)

        -- restore sample hooks
        localDataset.sampleHookTrain = localDataset.defaultSampleHook
        localDataset.sampleHookTest = localDataset.defaultSampleHook
    else -- single threaded data loading. useful for debugging
        donkeys = {}
        function donkeys:addjob(f1, f2) f2(f1()) end
        function donkeys:synchronize() end
        function donkeys:terminate() end
    end
end
