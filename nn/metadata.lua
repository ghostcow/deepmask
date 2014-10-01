require 'options'

----------------------------------------------------------------------
-- use -visualize to show network
-- parse command line arguments
if not opt then
    print '==> processing options'
    opt = getOptions()
end

local data_file = '../data_files/aligned/chunk_7k/cfw_flat'
numPersons = 559

-- classes - define classes array (used later for computing confusion matrix)
classes = {}
for i=1,numPersons do
    table.insert(classes, tostring(i))
end

function getDatasetFile(datasetPaths, iChunk)
	print(datasetPaths)
	print(iChunk)

	X = torch.load(datasetPaths[iChunk])
	local numImages = X.data:size()[4]
	local dataset = {
		-- the original matlab format is nImages x 3 x height x width
		-- (where height=width=152)
		-- but it's loaded into torch like this : width x height x 3 x nImages
		data = X.data:transpose(1,4):transpose(2,3),
		labels = X.labels[1],
		size = function() return numImages end
		}
	return dataset
end

-- convert to our general dataset format
trainDatasetPaths = {data_file..'_train_1.t7',data_file..'_train_2.t7',data_file..'_train_3.t7',data_file..'_train_4.t7',data_file..'_train_5.t7'}
testDatasetPaths = {data_file..'_test_1.t7',data_file..'_test_2.t7',data_file..'_test_3.t7'}

-- TODO - for some reason getDatasetFile working only with second argument equal to 1...
trainData = {
	numChunks = 1,
}
function trainData.getChunk(iChunk)
	return getDatasetFile(trainDatasetPaths, iChunk) 
end
testData = {
	numChunks = 1,
}
function testData.getChunk(iChunk)
	return getDatasetFile(testDatasetPaths, iChunk) 
end
