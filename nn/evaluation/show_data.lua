require 'gfx.js'
require 'options'

opt = getOptions()
dofile 'data.lua'
imageDim = 152

------------- samples per class
if (require 'gnuplot') then
    gnuplot.figure(1)
    gnuplot.hist(trainDataInner.labels, trainDataInner.labels:max())
    gnuplot.title('#samples per class - training')
    gnuplot.figure(2)
    gnuplot.hist(testDataInner.labels, testDataInner.labels:max())
    gnuplot.title('#samples per class - test')
end

hist = torch.Tensor(nLabels):fill(0)
for iImage = 1,trainDataInner:size() do
    label = trainDataInner.labels[iImage]
    hist[label] = hist[label] + 1
end
print(hist)

class_weights = torch.Tensor(nLabels):fill(0)
for iClass = 1,nLabels do
    class_weights[iClass] = 1 / hist[iClass]
end
class_weights = class_weights / class_weights:sum()
print('class weights in cost function :')
print(class_weights)

------------- show some
nImages = 100
for iSet = 1,2 do
    if (iSet == 1) then
        dataset = trainDataInner
        setName = 'train'
    else
        dataset = testDataInner
        setName = 'test'
    end

    for iClass = 1,1 do --nLabels do
        print(setName)
        print(dataset.size())
        print('==> random images per class '..tostring(iClass))
        dataSamples = torch.Tensor(nImages, 3, imageDim, imageDim)
        nTotalImages = dataset:size()
        shuffle = torch.linspace(1, nTotalImages, nTotalImages) -- torch.randperm(nTotalImages)
        i = 1
        for iImage = 1,nTotalImages do
            label = dataset.labels[shuffle[iImage]]
            if (label == iClass) then
                dataSamples[i] = dataset.data[shuffle[iImage]]
                i = i + 1
            end
            if (i > nImages) then
                break
            end
        end
        gfx.image(dataSamples, {legend=tostring(iClass)..' - '..setName})
    end
end



