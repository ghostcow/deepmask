-- balance dataset by down-sampling the common classes

require 'options'
opt = getOptions()
outputPath = string.sub(opt.dataPath, 1, -4)..'_balanced.t7'
dofile 'data.lua'
imageDim = 152

-- compute classes frequencies
classFreq = torch.Tensor(nLabels):fill(0)
for iImage = 1,trainDataInner:size() do
    label = trainDataInner.labels[iImage]
    classFreq[label] = classFreq[label] + 1
end
print(classFreq)
classSize = classFreq:min()

-- start collecting images into the new balanced dataset
datasetSizeNew = nLabels*classSize
trainNew = torch.Tensor(datasetSizeNew, 3, imageDim, imageDim)
trainLabelsNew = torch.Tensor(1, datasetSizeNew)
classFreqNew = torch.Tensor(nLabels):fill(0)
shuffle = torch.randperm(trainDataInner:size())
iImageNew = 1
for i = 1,trainDataInner:size() do
    iImage = shuffle[i]
    label = trainDataInner.labels[iImage]
    if (classFreqNew[label] < classSize) then
	print(iImageNew)
        trainNew[iImageNew] = trainDataInner.data[iImage]           
        trainLabelsNew[{1, iImageNew}] = label
        classFreqNew[label] = classFreqNew[label] + 1
	iImageNew = iImageNew + 1
    end
end

-- transpose images to get the original matlab format
trainNew = trainNew:transpose(1,4):transpose(2,3)
-- test ser is not changed but should be formatted to original dimensions
testNew = testDataInner.data:transpose(1,4):transpose(2,3)
testLabelsNew = torch.Tensor(1, testDataInner.size())
for iImage = 1,testDataInner.size() do
    testLabelsNew[{1, iImage}] = testDataInner.labels[iImage]
end

print('saving balanced dataset to '..outputPath)
torch.save(outputPath, {train = trainNew, trainLabels = trainLabelsNew, test = testNew, testLabels = testLabelsNew})
