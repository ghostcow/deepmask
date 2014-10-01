require 'options'

----------------------------------------------------------------------
-- use -visualize to show network
-- parse command line arguments
if not opt then
    print '==> processing options'
    opt = getOptions()
end

local data_file

if opt.size == 'small' then
  print '==> loading small dataset'
  data_file = '../data_set/cfw/cfw_small'
  numPersons = 200
elseif opt.size == 'full' then
  print '==> loading full dataset'
  -- this dataset contain only persons with 30-80 training samples
  data_file = '../data_files/aligned/cfw_flat'
  numPersons = 559
else
    error('unsupprted size option : '..opt.size)
end

-- classes - define classes array (used later for computing confusion matrix)
classes = {}
for i=1,numPersons do
  table.insert(classes, tostring(i))
end

local data_set
if (opt.dataFormat == 'mat') then
    require 'mattorch'
    -- look for mat file
    data_set = mattorch.load(data_file..'.mat')
elseif (opt.dataFormat == 't7') then
    -- look for torch file
    data_set = torch.load(data_file..'.t7')
else
    error('unsupprted dataFormat option : '..opt.dataFormat)
end

trsize = data_set.train:size()[4]
tesize = data_set.test:size()[4]
trainDataInner = {
	-- the original matlab format is nImages x 3 x height x width 
	-- (where height=width=152)
	-- but it's loaded into torch like this : width x height x 3 x nImages

	data = data_set.train:transpose(1,4):transpose(2,3),
	labels = data_set.trainLabels[1],
	size = function() return trsize end
	}

testDataInner = {
	data = data_set.test:transpose(1,4):transpose(2,3),
	labels = data_set.testLabels[1],
	size = function() return tesize end
}

-- convert to our general dataset format
trainData = {
	numChunks = 1,
	getChunk = function(iChunk) return trainDataInner end
}
testData = {
	numChunks = 1,
	getChunk = function(iChunk) return testDataInner end
}

------preprocessing - ?

------visualizing data---------------------------
if opt.visualize then
  require 'gfx.js'
  print '==> visualizing data'
  if (require 'gnuplot') then
	  gnuplot.figure(1)
	  gnuplot.hist(trainDataInner.labels, trainDataInner.labels:max())
	  gnuplot.title('#samples per person - training')
	  gnuplot.figure(2)
	  gnuplot.hist(testDataInner.labels, testDataInner.labels:max())
	  gnuplot.title('#samples per person - test')
  end

  local first100Samples_train = trainDataInner.data[{ {1,100} }]
  gfx.image(first100Samples_train, {legend='train - 100 samples'})
  local first100Samples_test = testDataInner.data[{ {1,100} }]
  gfx.image(first100Samples_test, {legend='test - 100 samples'})
end
