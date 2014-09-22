datasetFileName = 'cfw_flat_1001_1440.mat';
load(datasetFileName);
if true
    originalLabels = unique(trainLabels);
    offset = 0;
else
    offset = 240; % for cfw_flat_575_1000
end
maxNewLabel = max(originalLabels);
iNewLabel = maxNewLabel + 1;

for iPerson = originalLabels'
    fprintf('%d --> %d\n', iPerson, iNewLabel);
    trainLabels(trainLabels == iPerson) = iNewLabel;
    testLabels(testLabels == iPerson) = iNewLabel;
    
    iNewLabel = iNewLabel + 1;
end
trainLabels = trainLabels - maxNewLabel + offset;
testLabels = testLabels - maxNewLabel + offset;
save(datasetFileName, 'train', 'trainLabels', 'test', 'testLabels', 'originalLabels', '-v7.3');