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
    cmd:option('-visualize', false, 'visualize input data and weights during training')
    cmd:text()
    opt = cmd:parse(arg or {})
end

labels = torch.Tensor(1,1000)
labels:fill(1)
data = {
  train = torch.rand(152,152,3,1000),
  test = torch.rand(152,152,3,1000),
  trainLabels = labels,
  testLabels = labels 
}
trsize = data.train:size()[4]
tesize = data.test:size()[4]

numPersons = 200

trainData = {
  -- the original matlab format is nImages x 3 x height x width (where height=width=152)
  -- but it's loaded into torch like this : width x height x 3 x nImages
  
  data = data.train:transpose(1,4):transpose(2,3),
  labels = data.trainLabels[1],
  size = function() return trsize end
}

testData = {
  data = data.test:transpose(1,4):transpose(2,3),
  labels = data.testLabels[1],
  size = function() return tesize end
}

-- free some memory...
data = nil

------preprocessing - ?

------visualizing data---------------------------
if opt.visualize then
  print '==> visualizing data'
  first100Samples_train = trainData.data[{ {1,100} }]
  gfx.image(first100Samples_train, {legend='train - 100 samples'})
  first100Samples_test = testData.data[{ {1,100} }]
  gfx.image(first100Samples_test, {legend='test - 100 samples'})
end