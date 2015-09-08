require 'options'
require 'paths'
package.path = package.path .. ";" .. '../nn_utils/?.lua'

----------------------------------------------------------------------
print '==> processing options'
opt = getOptions()
-- nb of threads and fixed seed (for repeatable experiments)
torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor')

----------------------------------------------------------------------
if opt.gpu ~= -1 then
    require 'cutorch'
    cutorch.setDevice(opt.gpu)
end

----------------------------------------------------------------------
print '==> executing all'
dofile 'data.lua'
paths.dofile('model.lua')
paths.dofile('train.lua')
paths.dofile('test.lua')

----------------------------------------------------------------------
print '==> training!'

epoch = 1
test()

while epoch <= opt.epochs and not opt.testOnly do
    train()
    test()
end
