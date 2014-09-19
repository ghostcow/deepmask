require 'mattorch'
require 'gfx.js'

----------------------------------------------------------------------
-- use -visualize to show network
-- parse command line arguments
if not opt then
    print '==> processing options'
    cmd = torch.CmdLine()
    cmd:text()
    cmd:text('data for Deepface torch7 model')
    cmd:text()
    cmd:text('Options:')
    cmd:option('-size', 'small', 'how many samples do we load: small | full')
    cmd:option('-visualize', false, 'visualize input data and weights during training')
    cmd:text()
    opt = cmd:parse(arg or {})
end

if opt.size == 'small' then
  numPersons = 200
  local data_file = '../data_set/cfw/cfw_small.mat'
  local data_set = mattorch.load(data_file)
  trsize = data_set.train:size()[4]
  tesize = data_set.test:size()[4]
  trainData = {
    -- the original matlab format is nImages x 3 x height x width (where height=width=152)
    -- but it's loaded into torch like this : width x height x 3 x nImages
    
    data = data_set.train:transpose(1,4):transpose(2,3),
    labels = data_set.trainLabels[1],
    size = function() return trsize end
  }
  
  testData = {
    data = data_set.test:transpose(1,4):transpose(2,3),
    labels = data_set.testLabels[1],
    size = function() return tesize end
  }
elseif opt.size == 'full' then
  print('not implemented yet')
end

------preprocessing - ?

------visualizing data---------------------------
if opt.visualize then
  print '==> visualizing data'
  first100Samples_train = trainData.data[{ {1,100} }]
  gfx.image(first100Samples_train, {legend='train - 100 samples'})
  first100Samples_test = testData.data[{ {1,100} }]
  gfx.image(first100Samples_test, {legend='test - 100 samples'})
end