require 'gfx.js'
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
  -- beacause of limited memory the full 'cfw_flat.mat' cannot be loaded
  data_file = '../data_set/cfw/cfw_flat_1_574'
  numPersons = 240
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
    -- mattorch is installed --> look for mat file
    data_set = mattorch.load(data_file..'.mat')
elseif (opt.dataFormat == 't7') then
    -- mattorch is not installed --> look for torch file
    data_set = torch.load(data_file..'.t7')
else
    error('unsupprted dataFormat option : '..opt.dataFormat)
end

trsize = data_set.train:size()[4]
tesize = data_set.test:size()[4]
trainData = {
	-- the original matlab format is nImages x 3 x height x width 
	-- (where height=width=152)
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

------preprocessing - ?

------visualizing data---------------------------
if opt.visualize then
  print '==> visualizing data'
  if (require 'gnuplot') then
	  gnuplot.figure(1)
	  gnuplot.hist(trainData.labels, trainData.labels:max())
	  gnuplot.title('#samples per person - training')
	  gnuplot.figure(2)
	  gnuplot.hist(testData.labels, testData.labels:max())
	  gnuplot.title('#samples per person - test')
  end

  local first100Samples_train = trainData.data[{ {1,100} }]
  gfx.image(first100Samples_train, {legend='train - 100 samples'})
  local first100Samples_test = testData.data[{ {1,100} }]
  gfx.image(first100Samples_test, {legend='test - 100 samples'})
end
