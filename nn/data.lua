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
-- For the data-loading details, look at CocoDataLoader.lua

do -- start K datathreads (workers)
    if opt.nWorkers > 0 then
        -- make an upvalue to serialize over to worker threads
        local options = opt

        workers = Threads(opt.nWorkers,
            function()
                package.path = package.path .. ";" .. 'toolbox/?.lua' .. ";" .. '../nn_utils/?.lua'
                require 'torch'
                require 'cutorch'
                require 'CocoDataLoader'
                require 'math'
            end,

            function(idx)
                -- pass to all workers via upvalue
                opt = options
                dataset = torch.CocoDataLoader{splitName=opt.splitName,
                                               dataPath=opt.dataPath,
                                               imageDirPath=opt.imageDirPath,
                                               negativeRatio=opt.negativeRatio}
                tid = idx

                -- init thread seed
                local seed = opt.seed + idx
                torch.manualSeed(seed)
                cutorch.manualSeed(seed)
                print(string.format('Starting worker with id: %d seed: %d', tid, seed))
            end)
    else -- single threaded data loading. useful for debugging
        workers = {}
        function workers:addjob(f1, f2) f2(f1()) end
        function workers:synchronize() end
        function workers:terminate() end
    end
end
