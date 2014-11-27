% QUESTIONS :
% 1. what are targetPeopleX,targetPeopleSplitId, targetPeopleY ? (relevant only when RESTRICTED = false)
%
% guess -
% targetPeopleX is the people.txt data (unrestricted) with the appropriate
% targetPeopleSplitId/targetPeopleY indicating split id and labels
clear variables; close all; clc;
addpath('../../liblinear-1.94/matlab');
addpath('tools');

lfwDir = '/media/data/datasets/LFW';
pairsFilePath = fullfile(lfwDir, '/view2/pairs.txt');
peopleFilePath = fullfile(lfwDir, '/view2/people.txt');
type = 'deepid'; % type od the face images : deepface / deepid

%% constants 
% lfw configuration : restricted/unrestrcited
RESTRICTED = false; 
% should update the in-domain trained jb model, using the new domain data
updateInDomainJbModel = true; % original value = true
svmParams = struct('type', 1, 'C', 1); % original values : type=3, C=0.05
% finalTestMethod : 
% 1 = liblinear train & predict, 2 = liblinear train & our predict, 3 = choosing best threshold based on training
finalTestMethod = 3; 
maxNumNets = 15; % we can test accuracy if we use only some of the nets

if strcmp(type, 'deepface')
    lfwPeopleImagesFilePath = '../data_files/LFW/people.mat'; % relevant only when RESTRICTED = false
elseif strcmp(type, 'deepid')
    lfwPeopleImagesFilePath = '../data_files/deepId_full/LFW/people.mat'; % relevant only when RESTRICTED = false
end

[resDir, lfwpairsResFileName, lfwpeopleResFileName, verificationResFileName, verificationImagesFilePath] = ...
    GetResultFilePaths(3);

recognitionResDir = fullfile(resDir, 'recognition');
if ~exist(recognitionResDir, 'dir')
    mkdir(recognitionResDir)
end
recognitionResPath = fullfile(recognitionResDir, 'LFW_verification_results.txt');

%% loading Verification data (source domain)
% sourceX - data, sourceY - labels
fid = fopen(verificationImagesFilePath);
C = textscan(fid, '%s %d', 'Delimiter', ',');
fclose(fid);
verificationFeaturesLabels = C{2};

% loading verification set features
verificationResFiles = dir(fullfile(resDir, verificationResFileName));
nFiles = length(verificationResFiles);
nFiles = min(nFiles, maxNumNets);
% final 2D array with pairs feature, each with dimensions [featureDim x nPairs]
verificationFeatures = [];
for iFile = 1:nFiles
    S = load(fullfile(resDir, verificationResFiles(iFile).name));
    if isempty(verificationFeatures)
        subFeatureDim = size(S.x, 1);
        featureDim = nFiles*subFeatureDim;
        verificationFeatures = zeros(featureDim, length(verificationFeaturesLabels));
    end
    
    subFeatureIndex = 1 + (iFile - 1)*subFeatureDim;
    verificationFeatures(subFeatureIndex:(subFeatureIndex + subFeatureDim - 1), :) = S.x;
end
sourceX = verificationFeatures';
sourceY = verificationFeaturesLabels';

%% Loading LFW data (target domain)
[targetX1, targetX2, targetY, targetSplitId] = LoadLfwPairs(...
    fullfile(resDir, lfwpairsResFileName), maxNumNets);
if ~RESTRICTED
    % load by people (based on people.txt)
    [targetPeopleX, targetPeopleY, targetPeopleSplitId] = LoadLfwPeople(...
        fullfile(resDir, lfwpeopleResFileName), lfwPeopleImagesFilePath, maxNumNets);
end

%% PCA & joint-bayesian model learning over the source domain
outputDim = 150; %first parameter I tried
pcaFilePath = fullfile(recognitionResDir, 'pca_verification.mat');
fprintf('\nPCA over verification set\n');
if exist(pcaFilePath, 'file')
    load(pcaFilePath);
else
    [sourceXpca, sourcePcaModel] = wpca(sourceX,[],outputDim,[]);
    save(pcaFilePath, 'sourceXpca', 'sourcePcaModel');
end

jbFilePath = fullfile(recognitionResDir, 'jb_verification.mat');
fprintf('\njoint-bayesian over verification set\n');
if exist(jbFilePath, 'file')
    load(jbFilePath);
else  
    sourceJbModel = jointBayesian(sourceXpca',sourceY',[]);
    save(jbFilePath, 'sourceJbModel');
end

jbParams.initial_guess.Se = sourceJbModel.Se;
jbParams.initial_guess.Sm = sourceJbModel.Sm;
jbParams.transfer_model.Se = sourceJbModel.Se;
jbParams.transfer_model.Sm = sourceJbModel.Sm;
jbParams.transfer_model.lambda = 0.5;
jbParams.transfer_model.w = jbParams.transfer_model.lambda/(1+jbParams.transfer_model.lambda);

%% PCA projection over the target data
fprintf('\napplying pca over test data\n');
targetX1pca = wpca(targetX1',sourcePcaModel,outputDim,[]);
targetX2pca = wpca(targetX2',sourcePcaModel,outputDim,[]);
if ~RESTRICTED
    targetPeopleXpca = wpca(targetPeopleX',sourcePcaModel,outputDim,[]);
end

%% cross-validation over the target data
% for each split the joint bayesian model is updated using the training data
targetJbModels = cell(1);
jbScores = [];
accuracies = [];

lambdarange = [1.35]; % not the first parameter I tried
fprintf('\nstart testing...\n');
for i = [5,setdiff(1:10,5)]
    disp(i);
    if ~RESTRICTED
        ii = find(targetPeopleSplitId~=i);
        targetPeopleXpcaSplit = targetPeopleXpca(ii,:);
        [~,~,targetPeopleYsplit] = unique(targetPeopleY(ii));
    end
    targetIndicesTet = find(targetSplitId==i);
    targetIndicesTrain = find(targetSplitId~=i);
    
    for k = 1:length(lambdarange),
        thislambda = lambdarange(k);
        jbParams.transfer_model.lambda = thislambda;
        % weight factor when combining current model and new model
        jbParams.transfer_model.w = jbParams.transfer_model.lambda/(1+jbParams.transfer_model.lambda);
        % using the train folds to get new estimation for the joint-bayesian model
        if ~exist('targetJbModels','var') || size(targetJbModels,2)<i || size(targetJbModels,1)<k || isempty(targetJbModels{k,i})
            if updateInDomainJbModel
                jbFilePath = fullfile(recognitionResDir, sprintf('jb_lambda%0.2f_%d.mat', thislambda, i));
                if exist(jbFilePath, 'file')
                    load(jbFilePath, 'x');
                    targetJbModels{k,i} = x;
                else                    
                    if ~RESTRICTED
                        targetJbModels{k,i} = ...
                            jointBayesian(targetPeopleXpcaSplit',targetPeopleYsplit,jbParams);
                    else
                        posi = targetIndicesTrain(find(targetY(targetIndicesTrain)>0));
                        targetX1pcaSplit = [targetX1pca(posi,:);targetX2pca(posi,:)];
                        targetYsplit = [1:length(posi), 1:length(posi)];
                        targetJbModels{k,i} = ...
                            jointBayesian(targetX1pcaSplit', targetYsplit, jbParams);
                    end
                    x = targetJbModels{k,i};
                    save(jbFilePath, 'x');
                end
            else
                % use the existing model
                targetJbModels{k,i} = sourceJbModel;
            end
            
            jbScores(i,:,k) = ...
                jointBayesianC(targetX1pca', targetX2pca', targetJbModels{k,i})';
            
            trainData = double(jbScores(i,targetIndicesTrain,k));
            trainLabels = targetY(targetIndicesTrain); % TODO: originally it was y - is it the same ??
            testData = double(jbScores(i,targetIndicesTet,k));
            testLabels = targetY(targetIndicesTet);
            if (finalTestMethod ~= 3)
                MODEL = CLSliblinear(sparse(trainData), trainLabels, svmParams);
            end
            switch finalTestMethod
                case 1
                    weights = MODEL.svmmodel.w(1:end-1);
                    bias = MODEL.svmmodel.w(end);
                    trainRes = [weights bias] * [trainData; ones(1, size(trainData, 2))];
                    wasFlipped = mean(sign(trainRes) == trainLabels') < 0.5;
                    if wasFlipped
                        trainRes = -trainRes;
                    end
                    trainAccuracy = mean(sign(trainRes) == trainLabels');

                    % test
                    testRes = [weights bias] * [testData; ones(1, size(testData, 2))];
                    if wasFlipped
                        testRes = -testRes;
                    end
                    accuracies(k,i) = mean(sign(testRes) == testLabels')
                case 2
                    [ry,crwjbtl{i,k}] = CLSliblinearC(testData, MODEL);
                    %[~ ,crwjbtltrain{i,k}] = CLSliblinearC(double(jbScores(i,targetIndicesTrain,k)), MODEL);
                    accuracies(k,i) = mean(ry == testLabels)
                case 3
                    % the data is 1D, so all we need to find is the best TH
                    thValues = min(trainData):max(trainData);
                    numPos = sum(trainLabels == 1);
                    numNeg = sum(trainLabels == -1);
                    scoresPosPdf = histc(trainData(trainLabels == 1), thValues);
                    scoresNegPdf = histc(trainData(trainLabels == -1), thValues);

                    % scoresPosCdf(i) - how much positive examples have score <= thValues(i+1)
                    scoresPosCdf = cumsum(scoresPosPdf);
                    % scoresNegCdf(i) - how much negative examples have score <= thValues(i+1)
                    scoresNegCdf = cumsum(scoresNegPdf);

                    % compute accuracy per th, and choose best one
                    % correct detection : 
                    % positive with score > th or 
                    % negative with score < th
                    accuracyPerTh = (numPos - scoresPosCdf + scoresNegCdf)/(numPos + numNeg);
                    [maxAccuracy, thIndex] = max(accuracyPerTh);
                    bestTh = thValues(thIndex+1);

                    matchDetections = testData > bestTh;
                    matchDetections = 2*(matchDetections - 0.5);
                    accuracies(k,i) = mean(matchDetections == testLabels')
            end
        end
    end
end
accuracies
[mean(accuracies(1,:)) std(accuracies(1,:))]
dlmwrite(recognitionResPath, accuracies, '-append');
dlmwrite(recognitionResPath, [mean(accuracies(1,:)) std(accuracies(1,:))], '-append');