require 'randomize_pairs'
data = torch.load('../../data_files/CFW_small/cfw_small.t7')
labels = data.trainLabels[{1}]

shuffle = RandomizePairs.randomizeImages(data, labels)
numPosPairs = 0
numNegPairs = 0
for iImage=1,shuffle:size(1),2 do
    label1 = labels[shuffle[iImage]]
    label2 = labels[shuffle[iImage+1]]
    if (label1 == label2) then
        numPosPairs = numPosPairs + 1
        pairLabel = 1
    else
        numNegPairs = numNegPairs + 1
        pairLabel = -1
    end
    print(pairLabel, label1, label2)
end

print('nImages=',labels:size(1))
print('numPosPairs=',numPosPairs)
print('numNegPairs=',numNegPairs)
print('numPairs=',numPosPairs+numNegPairs)