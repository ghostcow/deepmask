clear all variables; close all; clc;
addpath('../../libsvm-3.19/matlab');
addpath('../../liblinear-1.95/matlab');
addpath('lior');

%% parameters
useNormalization = true;
p = 2; % which p to use in Lp normalizarion
numFolds = 10;
numPairsPerFold = 300;
classifierType = 'liblinear'; % options : libsvm / liblinear
mode = 1; % 1 - restricted image configuration, 2 - unrestricted configuration

% relvecant only for mode = 2, limit the number of pairs taken for the svm
% training (for each of the 2 classes)
numPairsForTrain = 100000;
%% load data
LoadPairsData;

if (mode == 2)
    peopleMetadata = GetPeopleData();
    peopleFeatures = load('lfw_people.mat');
    peopleFeatures = peopleFeatures.x;
end

%% SVM classifying, cross-validation
accuracies = zeros(1, numFolds);
for iTestFold = 1:numFolds
    testFoldStartIndex = 1 + (iTestFold - 1)*2*numPairsPerFold;
    
    testPairIndices = testFoldStartIndex:(testFoldStartIndex + 2*numPairsPerFold - 1);
    testLabels = labels(testPairIndices);
    
    if (mode == 1)
        % restricted image configuration
        trainPairIndices = setdiff(1:nPairs, testPairIndices);
        trainLabels = labels(trainPairIndices);
        trainLeft = datasetLeftFace(:, trainPairIndices);
        trainRight = datasetRightFace(:, trainPairIndices);
    elseif (mode == 2)
        % unrestricted configuration
        trainFoldsIndices = setdiff(1:numFolds, iTestFold);
        
        pairsFilePath = sprintf('lfw_pairs_unrestricted_%d.mat', iTestFold);
        if ~exist(pairsFilePath, 'file')
            [trainPairIndices, trainLabels] = GetAllPairs(cell2mat(peopleMetadata(trainFoldsIndices)));
        	save(pairsFilePath, 'trainPairIndices', 'trainLabels');
        else
            load(pairsFilePath, 'trainPairIndices', 'trainLabels');
        end
        locPos = find(trainLabels == 1);
        locNeg = find(trainLabels == -1);        
        numPos = sum(locPos);
        numNeg = sum(locNeg);
        numToTake = min([numPairsForTrain, numPos, numNeg]);

        randIndices = randperm(length(locNeg)); randIndices = randIndices(1:numToTake); locNeg = locNeg(randIndices);
        randIndices = randperm(length(locPos)); randIndices = randIndices(1:numToTake); locPos = locPos(randIndices);
        
        trainPairIndices = [trainPairIndices(locPos, :); trainPairIndices(locNeg, :)];
        trainLabels = [trainLabels(locPos), trainLabels(locNeg)]';
        
        trainLeft = peopleFeatures(:, trainPairIndices(:, 1));
        trainRight = peopleFeatures(:, trainPairIndices(:, 2));
    end
    
    %% normalization
    % transforming pair indices to features indices (for example pair 1
    % trasnfomrs into indices 1,2. pair 2 into 3,4. etc.)

    if useNormalization
        % 1. compute normalization factors based on training data
        normFactors1 = max(trainLeft, [], 2);
        normFactors2 = max(trainRight, [], 2);
        normFactors =  max([normFactors1 normFactors2], [], 2);
        normFactors(normFactors == 0) = inf;
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
        classifier = train(trainLabels, sparse(chiSquaredDistsTrain'), '-B 1'); % -c 0.05 -s 1');
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