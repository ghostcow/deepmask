require 'mattorch'
require 'gfx.js'

data_file = '../data_set/cfw/cfw_small.mat'
data = mattorch.load(data_file)
trsize = data.train:size()[4]
tesize = data.test:size()[4]
numPerosns = 200

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

------visualizing data
----------------------------------------------------------------------
print '==> visualizing data'
first100Samples_train = trainData.data[{ {1,100} }]
gfx.image(first100Samples_train, {legend='train - 100 samples'})
first100Samples_test = testData.data[{ {1,100} }]
gfx.image(first100Samples_test, {legend='test - 100 samples'})