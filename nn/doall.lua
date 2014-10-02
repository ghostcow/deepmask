require 'options'

----------------------------------------------------------------------
print '==> processing options'
opt = getOptions()

-- nb of threads and fixed seed (for repeatable experiments)
torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)

----------------------------------------------------------------------
print '==> executing all'

if opt.useDatasetChunks then
    print '==> datasets is read in chunks'
    dofile 'metadata.lua'
else
    dofile 'data.lua'
end

dofile 'model.lua'
dofile 'train.lua'
dofile 'test.lua'

----------------------------------------------------------------------
print '==> training!'

while true do
    train()
    test()
end
