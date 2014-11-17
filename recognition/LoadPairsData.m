testData = load('testData.mat'); testData = testData.x;
testData = shiftdim(testData, 2);
testLabels = load('testLabels.mat'); testLabels = testLabels.x;

trainData = load('trainData.mat'); trainData = trainData.x;
trainData = shiftdim(trainData, 2);
trainLabels = load('trainLabels.mat'); trainLabels = trainLabels.x;

% concat train & test into one array
labels = [testLabels; trainLabels];
dataset = cat(1, testData, trainData);
[nPairs, featureDim, ~] = size(dataset);
% stack all features into one 2D array with dimensions [featureDim x 2*nPairs]
dataset = shiftdim(dataset, 1); 
dataset = reshape(dataset, [featureDim, 2*nPairs]);
% final 2D array with pairs feature, each with dimensions [featureDim x nPairs]
datasetLeftFace = dataset(:, 1:2:(2*nPairs));
datasetRightFace = dataset(:, 2:2:(2*nPairs));