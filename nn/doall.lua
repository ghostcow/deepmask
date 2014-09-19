

----------------------------------------------------------------------
print '==> processing options'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Deepface training')
cmd:text()
cmd:text('Options:')
-- global:
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-threads', 2, 'number of threads')
cmd:option('-visualize', false, 'visualize input data and weights during training')
-- data:
cmd:option('-size', 'full', 'how many samples do we load: small | full | extra')
-- training:
-- TODO: update to defaults values to match deepface training params
cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
cmd:option('-plot', false, 'live plot')
cmd:option('-learningRate', 0.01, 'learning rate at t=0')
cmd:option('-batchSize', 128, 'mini-batch size (1 = pure stochastic)')
cmd:option('-weightDecay', 0, 'weight decay for SGD')
cmd:option('-momentum', 0.9, 'momentum for SGD')
cmd:text()
opt = cmd:parse(arg or {})

-- nb of threads and fixed seed (for repeatable experiments)
torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)

----------------------------------------------------------------------
print '==> executing all'

dofile 'data.lua'
dofile 'model.lua'
dofile 'train.lua'
dofile 'test.lua'

----------------------------------------------------------------------
print '==> training!'

while true do
    train()
    test()
end