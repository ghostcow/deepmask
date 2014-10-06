require 'options'

----------------------------------------------------------------------
-- use -visualize to show network
-- parse command line arguments
if not opt then
    print '==> processing options'
    opt = getOptions()
end

nLabels = 559

-- classes - define classes array (used later for computing confusion matrix)
classes = {}
for i=1,numPersons do
    table.insert(classes, tostring(i))
end

function getDatasetFile(datasetPath)
    print('getDatasetFile: '..datasetPath)
	X = torch.load(datasetPath)
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
local data_file = '../data_files/aligned/cfw_flat'
trainData = {
	numChunks = 1, 
}
function trainData.getChunk(iChunk)
	return getDatasetFile(data_file..'_train_'..tostring(iChunk)..'.t7')
end
testData = {
	numChunks = 1,
}
function testData.getChunk(iChunk)
    return getDatasetFile(data_file..'_ttest_'..tostring(iChunk)..'.t7')
end
