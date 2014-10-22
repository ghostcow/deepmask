clear all; close all; clc;
addpath('../../../liblinear-1.94/matlab');

numPairsPerFold = 300;

%% load data
testData = load('../testData.mat'); testData = testData.x;
testData = shiftdim(testData, 2);
testLabels = load('../testLabels.mat'); testLabels = testLabels.x;

trainData = load('../trainData.mat'); trainData = trainData.x;
trainData = shiftdim(trainData, 2);
trainLabels = load('../trainLabels.mat'); trainLabels = trainLabels.x;

labels = [testLabels; trainLabels];
dataset = cat(1, testData, trainData);
[nPairs, featureDim, ~] = size(dataset);
dataset = shiftdim(dataset, 1); 
dataset = reshape(dataset, [featureDim, 2*nPairs]);

iTestFold = 1;
testFoldStartIndex = 1 + (iTestFold - 1)*2*numPairsPerFold;
   
testPairIndices = testFoldStartIndex:(testFoldStartIndex + 2*numPairsPerFold - 1);
trainPairIndices = setdiff(1:nPairs, testPairIndices);
testSize = length(testPairIndices);
trainSize = length(trainPairIndices);
    
trainLabels = labels(trainPairIndices);
testLabels = labels(testPairIndices);
% train data
trainDataLeftIndices = 1 + 2*(trainPairIndices - 1);
trainDataRightIndices = 2 + 2*(trainPairIndices - 1);
trainDataLeft = dataset(:, trainDataLeftIndices);
trainDataRight = dataset(:, trainDataRightIndices);

% test data
testDataLeftIndices = 1 + 2*(testPairIndices - 1);
testDataRightIndices = 2 + 2*(testPairIndices - 1);
testDataLeft = dataset(:, testDataLeftIndices);
testDataRight = dataset(:, testDataRightIndices);

% first normalization : divide by the maximum value for each feature element
    % TODO
    
%% conversion to lior variables names
LEFT = trainDataLeft;
RIGHT = trainDataRight;
LEFT_test = testDataLeft;
RIGHT_test = testDataRight;
SNS = trainLabels;
SNS_test = testLabels;
SNS(SNS == -1) = 0;
SNS_test(SNS_test == -1) = 0;
% TODO : maybe not all pairs were used for training
ii = 1:trainSize; 
jj = 1:testSize;
p = 2;
%% original lior code
LEFT        =normy2fast(LEFT, p);
RIGHT       =normy2fast(RIGHT, p);
LEFT_test   =normy2fast(LEFT_test, p);
RIGHT_test  =normy2fast(RIGHT_test, p);


X    =((LEFT      - RIGHT       ).^2) ./ (LEFT      + RIGHT      + 0.0001);
XTest=((LEFT_test - RIGHT_test  ).^2) ./ (LEFT_test + RIGHT_test + 0.0001);

%liblinear using C = 0.05

y       = 2*SNS     -1;
ytest   = 2*SNS_test-1;

C=0.05;

% TODO : not sure 3 means type parameter or regression
params = struct('C', C); % struct('type', 3, 'C', C); 

Model = CLSliblinear(sparse(double((X(:,ii)))),y(ii), params);  
weights = Model.svmmodel.w;
bias = Model.svmmodel.bias;
clsw = [weights bias]';
sc=[weights bias] * [(XTest(:,jj)); ones(1,length(jj))];
sctrain=[weights bias] * [(X); ones(1,size(X,2))];
wasflipped = mean(sign(sctrain)==y')<.5; 
if wasflipped
  sc = -sc; sctrain = -sctrain;
end

corrects = sign(sc)==ytest(jj)';
score = mean(corrects);
disp(score);