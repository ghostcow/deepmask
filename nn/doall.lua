require 'torch'
require 'options'
require 'paths'
package.path = package.path .. ";" .. 'toolbox/?.lua' .. ";" .. '../nn_utils/?.lua'

----------------------------------------------------------------------
print '==> processing options'
opt = getOptions()
-- nb of threads and fixed seed (for repeatable experiments)
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opt.seed)

----------------------------------------------------------------------
if opt.gpu ~= -1 then
    require 'cutorch'
    cutorch.setDevice(opt.gpu)
    cutorch.manualSeed(opt.seed)
end

----------------------------------------------------------------------
print '==> executing all'
dofile 'data.lua'
dofile 'model.lua'
dofile 'train.lua'
--paths.dofile('test.lua')

----------------------------------------------------------------------
print '==> training!'

epoch = 1
--test()

while epoch <= opt.epochs and not opt.testOnly do
    train()
--    test()
end
