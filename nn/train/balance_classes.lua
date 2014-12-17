function getClassWeights()
    print '==> balance classes in NLL'
    --- compute frequency for each class
    class_freqs = torch.Tensor(nLabels):fill(0)
    for iChunk = 1,trainData.numChunks do
        trainDataChunk = trainData.getChunk(iChunk)
        for iImage = 1,trainDataChunk:size() do
            label = trainDataChunk.labels[iImage]
            class_freqs[label] = class_freqs[label] + 1
        end
    end
    print('classes frequencies :')
    print(class_freqs)
    class_weights = torch.Tensor(nLabels):fill(0)
    for iLabel = 1,nLabels do
        class_weights[iLabel] = 1 / class_freqs[iLabel]
    end
    class_weights = class_weights / class_weights:sum()
    print('classes weights :')
    print(class_weights)
end