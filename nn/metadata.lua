require 'gfx.js'
require 'options'

local getDataset = function(metadata, iFile)

    local data
    local labels
    X = torch.load(metadata.paths(iFile))

    -- the original matlab format is nImages x 3 x height x width
    -- (where height=width=152)
    -- but it's loaded into torch like this : width x height x 3 x nImages
    data = X.data:transpose(1,4):transpose(2,3)
    labels = X.labels
    return data,labels
end

----------------------------------------------------------------------
-- use -visualize to show network
-- parse command line arguments
if not opt then
    print '==> processing options'
    opt = getOptions()
end

local data_file
data_file = '../data_set/cfw/aligned/cfw_flat'
numPersons = 558

-- classes - define classes array (used later for computing confusion matrix)
classes = {}
for i=1,numPersons do
    table.insert(classes, tostring(i))
end

trainMetaData = {
    paths = {data_file..'_train.1.t7', data_file..'_train.2.t7'},
    nFiles = 2
}

testMetaData = {
    paths = {data_file..'_test.1.t7'},
    nFiles = 1
}