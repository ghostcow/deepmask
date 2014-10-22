clear all; close all; clc;
addpath('../../libsvm-3.18/matlab');
addpath('../../liblinear-1.94/matlab');
addpath('lior');

%% parameters
useNormalization = true;
p = 2; % which p to use in Lp normalizarion
numFolds = 10;
numPairsPerFold = 300;
classifierType = 'liblinear'; % options : libsvm / liblinear

%% load data
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

%% compute chisquare distances
% chiSquaredDists = GetChiSquaredDists(dataset);

%% SVM classifying, cross-validation
accuracies = zeros(1, numFolds);
for iTestFold = 1:numFolds
    testFoldStartIndex = 1 + (iTestFold - 1)*2*numPairsPerFold;
    
    testPairIndices = testFoldStartIndex:(testFoldStartIndex + 2*numPairsPerFold - 1);
    trainPairIndices = setdiff(1:nPairs, testPairIndices);
    
    trainLabels = labels(trainPairIndices);
    testLabels = labels(testPairIndices);
    %% normalization
    % transforming pair indices to features indices (for example pair 1
    % trasnfomrs into indices 1,2. pair 2 into 3,4. etc.)
    trainLeft = datasetLeftFace(:, trainPairIndices);
    trainRight = datasetRightFace(:, trainPairIndices);
    if useNormalization
        % 1. compute normalization factors based on training data
        normFactors =  max([trainLeft trainRight], [], 2);
        trainLeft = bsxfun(@times, trainLeft, 1./normFactors);
        trainRight = bsxfun(@times, trainRight, 1./normFactors);
        % 2. L2 normalization
        trainLeft = normy2fast(trainLeft, p);
        trainRight = normy2fast(trainRight, p);
    end
 
    %% train
    disp('training svm classifier...');
    chiSquaredDistsTrain = ((trainLeft - trainRight).^2) ./ (trainLeft + trainRight + 0.0001);
    if strcmp(classifierType, 'libsvm')
        classifier = svmtrain(trainLabels, chiSquaredDistsTrain', '-t 0 -h 0');
        weights = sum(bsxfun(@times, classifier.sv_coef, classifier.SVs));
        bias = -classifier.rho;
    elseif strcmp(classifierType, 'liblinear')
        classifier = train(trainLabels, sparse(chiSquaredDistsTrain'), '-B 1'); % -s 3 -c 0.05 ?
        weights = classifier.w(1:end-1);
        bias = classifier.w(end);
    else
        error('unknown classifier type');
    end
    
    trainRes = [weights bias] * [chiSquaredDistsTrain; ones(1, size(chiSquaredDistsTrain, 2))];
    wasFlipped = mean(sign(trainRes) == trainLabels') < 0.5; 
    if wasFlipped
      trainRes = -trainRes;
    end
    trainAccuracy = mean(sign(trainRes) == trainLabels');
    
    %% test
    disp('testing...');
	testLeft = datasetLeftFace(:, testPairIndices);
    testRight = datasetRightFace(:, testPairIndices);
    if useNormalization
        % 1. use normalization factors based on training data
        testLeft = bsxfun(@times, testLeft, 1./normFactors);
        testRight = bsxfun(@times, testRight, 1./normFactors);
        % 2. L2 normalization
        testLeft = normy2fast(testLeft, p);
        testRight = normy2fast(testRight, p);
    end

    chiSquaredDistsTest = ((testLeft - testRight).^2) ./ (testLeft + testRight + 0.0001);
    testRes = [weights bias] * [chiSquaredDistsTest; ones(1, size(chiSquaredDistsTest, 2))];
    if wasFlipped
      testRes = -testRes; 
    end
    accuracies(iTestFold) = mean(sign(testRes) == testLabels');
    fprintf('Fold no. %d - accuracy = %f %% \n', iTestFold, accuracies(iTestFold));
end
fprintf('average accuracy = %f %% \n', mean(accuracies));