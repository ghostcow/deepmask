require 'gfx.js'
require 'options'

opt = getOptions()
dofile 'data.lua'
imageDim = 152

print '==> samples per person'
if (require 'gnuplot') then
    gnuplot.figure(1)
    gnuplot.hist(trainDataInner.labels, trainDataInner.labels:max())
    gnuplot.title('#samples per person - training')
    gnuplot.figure(2)
    gnuplot.hist(testDataInner.labels, testDataInner.labels:max())
    gnuplot.title('#samples per person - test')
end

nImages = 100
for iSet = 1,2 do
    if (iSet == 1) then
        dataset = trainDataInner
        setName = 'train'
    else
        dataset = testDataInner
        setName = 'test'
    end

    for iPerson = 2,2 do --numPersons do
        print('==> random images per person '..tostring(iPerson))
        dataSamples = torch.Tensor(nImages, 3, imageDim, imageDim)
        nTotalImages = dataset:size()
        shuffle = torch.linspace(1, nTotalImages, nTotalImages) -- torch.randperm(nTotalImages)
        i = 1
        for iImage = 1,nTotalImages do
            label = dataset.labels[shuffle[iImage]]
            if (label == iPerson) then
                dataSamples[i] = dataset.data[shuffle[iImage]]
                i = i + 1
            end
            if (i > nImages) then
                break
            end
        end
        gfx.image(dataSamples, {legend=tostring(iPerson)..' - '..setName})
    end
end



