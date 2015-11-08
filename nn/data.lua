require 'options'
require 'paths'
require 'CocoDataLoader'
local Threads = require 'threads'

----------------------------------------------------------------------
-- use -visualize to show network
-- parse command line arguments

if not opt then
    print '==> processing options'
    opt = getOptions()
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
                package.path = package.path .. ";" .. 'toolbox/?.lua'
                gsdl = require 'sdl2'
                require 'torch'
                require 'CocoDataLoader'
                require 'math'
            end,

            function(idx)
                -- pass to all donkeys via upvalue
                opt = options
                dataset = torch.dataLoader{dataPath=opt.dataPath,
                                            cocoImagePath=opt.cocoImagePath,
                                            negativeRatio=opt.negativeRatio}
                tid = idx

                -- init thread seed
                local seed = opt.seed + idx
                torch.manualSeed(seed)
                print(string.format('Starting donkey with id: %d seed: %d', tid, seed))
            end)
    else -- single threaded data loading. useful for debugging
        donkeys = {}
        function donkeys:addjob(f1, f2) f2(f1()) end
        function donkeys:synchronize() end
        function donkeys:terminate() end
    end
end
