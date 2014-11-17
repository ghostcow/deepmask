package.path = package.path .. ";../?.lua"
require 'options'

----------------------------------------------------------------------
print '==> processing options'
opt = getOptions()
opt.save = paths.concat('../../results_deepid/', opt.save)
-- nb of threads and fixed seed (for repeatable experiments)
torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)

----------------------------------------------------------------------
--------- write options table to log file ----------------------------
os.execute('mkdir -p ' .. opt.save)
logFilePath = paths.concat(opt.save, 'doall.log')
fid = io.open(logFilePath,"a")
fid:write(os.date("%Y_%m_%d_%X"), '\n')
fid:write("opt : {")
for filedName,fieldValue in pairs(opt) do
    fid:write(filedName, " = ", tostring(fieldValue), ', ')
end
fid:write("}\n\n")
fid:close()

----------------------------------------------------------------------
print '==> executing all'
dofile 'data_patch.lua'

dofile('../model_deepID.lua')
dofile('../train.lua')
if not opt.trainOnly then
    dofile('../test.lua')
end

--- check data validity before starting
-- data
for iChunk = 1,trainData.numChunks do
    trainDataChunk = trainData.getChunk(iChunk)
    -- shuffle at each epoch
    for t = 1,trainDataChunk:size(),opt.batchSize do
        -- create mini batch
        local inputs = torch.Tensor(opt.batchSize, 3, imageDim, imageDim)
        local targets = torch.Tensor(opt.batchSize)
        if ((t+opt.batchSize-1) > trainDataChunk:size()) then
            -- we don't use the last samples
            break
        end
        for i = t,(t+opt.batchSize-1) do
            inputs[{i-t+1}] = trainDataChunk.data[i]
            targets[{i-t+1}] = trainDataChunk.labels[i]
        end
        assert(isValid(targets), "non-valid targets")
        assert(isValid(inputs), "non-valid inputs")
        inputs = inputs:cuda()
        assert(isValid(inputs), "non-valid cuda inputs")
    end
end
-- model
assert(isValid(parameters), "non-valid model parameters")

----------------------------------------------------------------------
if not opt.debugOnly then
    print '==> training!'
    while true do
        train()
        if not opt.trainOnly then
            test()
        end
    end
end