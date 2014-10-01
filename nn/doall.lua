require 'options'

----------------------------------------------------------------------
print '==> processing options'
opt = getOptions()

-- nb of threads and fixed seed (for repeatable experiments)
torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)

----------------------------------------------------------------------
print '==> executing all'

dofile 'data.lua'
if not opt.loadState then
    dofile 'model.lua'
end
dofile 'train.lua'
dofile 'test.lua'

----------------------------------------------------------------------
print '==> training!'

while true do
    train()
    test()
end
