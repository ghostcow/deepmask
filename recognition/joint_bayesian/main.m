% QUESTIONS :
% 1. what are lfwX,trainsplit, trainlabel ? (relevant only when RESTRICTED = false)
%
% guess -
% lfwX is the people.txt data (unrestricted) with the appropriate
% trainsplit/trainlabel indicating split id and labels
clear variables; close all; clc;
addpath('..');
addpath('../../../liblinear-1.94/matlab');

lfwDir = '/media/data/datasets/LFW';
pairsFilePath = fullfile(lfwDir, '/view2/pairs.txt');
peopleFilePath = fullfile(lfwDir, '/view2/people.txt');

%% constants 
numFolds = 10;
numPairsPerFold = 300; % half of #pairs in each fold (positive/negative pairs)
nPairs = numFolds*numPairsPerFold*2;
jbTrainingSetType = 2; % 1 - use LFW data only, 2 - use verification set
% lfw configuration : restricted/unrestrcited
RESTRICTED = false; % original value = false
% should update the in-domain trained jb model, using the new domain data
updateInDomainJbModel = true; % original value = true
svmParams = struct('type', 1, 'C', 1); % original values : type=3, C=0.05
% finalTestMethod : 
% 1 = liblinear train & predict, 2 = liblinear train & our predict, 3 = choosing best threshold based on training
finalTestMethod = 1; 

lfwPeopleImagesFilePath = '../../data_files/LFW/people.mat'; % relevant only when RESTRICTED = false

if true
    resDir = '../../results_deepid/CFW_PubFig_SUFR_deepID.3.64_dropout_flipped';
    lfwpairsResFileName = 'deepid_LFW_pairs_patch*';
    lfwpeopleResFileName = 'deepid_LFW_people_patch*';
    verificationResFileName = 'deepid_CPS_verification_patch*';
    verificationImagesFilePath = '../../data/deepId/CFW_PubFig_SUFR/images_verification.txt';
else
    resDir = '../../results_deepid/CFW_PubFig_SUFR_deepID.3.64_dropout_flipped_ReLu';
    lfwpairsResFileName = 'LFW_pairs_patch*';
    lfwpeopleResFileName = 'LFW_people_patch*';
    verificationResFileName = 'verification_patch*';
    verificationImagesFilePath = '../../data/deepId/CFW_PubFig_SUFR/images_verification.txt';    
end

recognitionResDir = fullfile(resDir, 'recognition');
if ~exist(recognitionResDir, 'dir')
    mkdir(recognitionResDir)
end

%% Loading LFW pairs data (10 splits bundled together)
% loadind lfw pairs features
lfwpairsResFiles = dir(fullfile(resDir, lfwpairsResFileName));
nFiles = length(lfwpairsResFiles);
% final 2D array with pairs feature, each with dimensions [featureDim x nPairs]
datasetLeftFace = []; 
datasetRightFace = [];
for iFile = 1:nFiles
    S = load(fullfile(resDir, lfwpairsResFiles(iFile).name));
    if isempty(datasetLeftFace)
        subFeatureDim = size(S.x, 1);
        featureDim = nFiles*subFeatureDim;
        datasetLeftFace = zeros(featureDim, nPairs);
        datasetRightFace = zeros(featureDim, nPairs);
    end
    subFeatureIndex = 1 + (iFile - 1)*subFeatureDim;
    datasetLeftFace(subFeatureIndex:(subFeatureIndex + subFeatureDim - 1), :) = S.x(:, 1:2:end);
    datasetRightFace(subFeatureIndex:(subFeatureIndex + subFeatureDim - 1), :) = S.x(:, 2:2:end);
end

% convert lfw names to numbers (directories indices)
figDirs = dir(fullfile(lfwDir, 'lfw'));
figDirs = figDirs(3:end);
figDirs = {figDirs.name};
mapNameToNum = containers.Map;
for iDir = 1:length(figDirs)
    mapNameToNum(figDirs{iDir}) = iDir;
end

if ~RESTRICTED
    % lfwX       - Nxd feature vectors for all face images (load from mat file)
    % trainlabel - person label for each face image
    % trainsplit - split id for each face image
    
    S = load(lfwPeopleImagesFilePath, 'labels', 'splitId');
    nImages = length(S.labels);
    trainlabel = S.labels;
    trainsplit = S.splitId;
    
    lfwpeopleResFiles = dir(fullfile(resDir, lfwpeopleResFileName));
    nFiles = length(lfwpeopleResFiles);
    % final 2D array with pairs feature, each with dimensions [featureDim x nPairs]
    lfwX = []; 
    for iFile = 1:nFiles
        S = load(fullfile(resDir, lfwpeopleResFiles(iFile).name));
        if isempty(lfwX)
            subFeatureDim = size(S.x, 1);
            featureDim = nFiles*subFeatureDim;
            lfwX = zeros(featureDim, nImages);
        end
        subFeatureIndex = 1 + (iFile - 1)*subFeatureDim;
        lfwX(subFeatureIndex:(subFeatureIndex + subFeatureDim - 1), :) = S.x(:, 1:end);
    end
end

if (jbTrainingSetType == 1)    
    % TEMP - pca & joint-baysian learning using the LFW data (eventually it
    % will be done with other external data)    
    % this method is illegal, because we are using all set set together
    [namesLeft, namesRight, imgNumLeft, imgNumRight] = ...
        ParsePairsFile(pairsFilePath);
    idsLeft = cellfun(@(x)(mapNameToNum(x)), namesLeft);
    idsRight = cellfun(@(x)(mapNameToNum(x)), namesRight);

    XSOURCEDOMAIN = [datasetLeftFace'; datasetRightFace'];
    ysource = [idsLeft, idsRight];
elseif (jbTrainingSetType == 2)
    % loading verification images labels 
    fid = fopen(verificationImagesFilePath);
    C = textscan(fid, '%s %d', 'Delimiter', ',');
    fclose(fid);
    verificationFeaturesLabels = C{2};
    
    % loading verification set features
    verificationResFiles = dir(fullfile(resDir, verificationResFileName));
    nFiles = length(lfwpairsResFiles);
    % final 2D array with pairs feature, each with dimensions [featureDim x nPairs]
    verificationFeatures = zeros(featureDim, length(verificationFeaturesLabels));
    for iFile = 1:nFiles
        S = load(fullfile(resDir, verificationResFiles(iFile).name));
        subFeatureIndex = 1 + (iFile - 1)*subFeatureDim;
        verificationFeatures(subFeatureIndex:(subFeatureIndex + subFeatureDim - 1), :) = S.x;
    end
    XSOURCEDOMAIN = verificationFeatures';
    ysource = verificationFeaturesLabels';
end

%% original code
%SPLITS has all data as separate cells.
%Make one big first image XX1 and second image XX2
%y is length 6000 at end and so is splitid

% test data (seperated into folds)
if false % original code, loading data from SPLITS cell array
    for i = 1:10,
        XX1 = [XX1,SPLITS{i}{4}];
        XX2 = [XX2,SPLITS{i}{5}];
        ylong  = [ylong; SPLITS{i}{6}*2-1];
        splitid = [splitid;ones(length(SPLITS{i}{6}),1)*i];
    end
    y = ylong(1:600); %for just one split
else
    XX1 = datasetLeftFace;
    XX2 = datasetRightFace;
    ylong = zeros(nPairs, 1);
    splitid = zeros(size(XX1, 2), 1);
    for iFold = 1:numFolds
        startIndex = 1 + (iFold-1)*2*numPairsPerFold;
        splitid(startIndex:(startIndex+2*numPairsPerFold-1)) = iFold;
        ylong(startIndex:(startIndex+numPairsPerFold-1)) = 1; % positive pairs
        ylong((startIndex+numPairsPerFold):(startIndex+2*numPairsPerFold-1)) = -1; % negative pairs
    end
end

%% PCA & joint-bayesian model learning, using extranal data
outputDim = 250; %first parameter I tried
pcaFilePath = fullfile(recognitionResDir, 'pca_verification.mat');
fprintf('\nPCA over verification set\n');
if exist(pcaFilePath, 'file')
    load(pcaFilePath);
else
    [XX_pca,INDOPCAModel] = WPCA_v2(XSOURCEDOMAIN,[],outputDim,[]);
    save(pcaFilePath, 'XX_pca', 'INDOPCAModel');
end

jbFilePath = fullfile(recognitionResDir, 'jb_verification.mat');
fprintf('\njoint-bayesian over verification set\n');
if exist(jbFilePath, 'file')
    load(jbFilePath);
else  
    INDOJBMODEL = get_EMLike_model_clean(XX_pca',ysource',[]);
    save(jbFilePath, 'INDOJBMODEL');
end

params.initial_guess.Se = INDOJBMODEL.Se;
params.initial_guess.Sm = INDOJBMODEL.Sm;
params.transfer_model.Se = INDOJBMODEL.Se;
params.transfer_model.Sm = INDOJBMODEL.Sm;
params.transfer_model.lambda = 0.5;
params.transfer_model.w = params.transfer_model.lambda/(1+params.transfer_model.lambda);

%% pca projection over the test data
fprintf('\napplying pca over test data\n');
XX1proj = WPCA_v2(XX1',INDOPCAModel,outputDim,[]);
XX2proj = WPCA_v2(XX2',INDOPCAModel,outputDim,[]);
if ~RESTRICTED
    XX_pcaall = WPCA_v2(lfwX',INDOPCAModel,outputDim,[]);
end

cPCAModel = cell(1);
cBJmodel = cell(1);

lambdarange = [1.35]; %not the first parameter I tried
fprintf('\nstart testing...\n');
for i = [5,setdiff(1:10,5)]
    disp(i);
    if ~RESTRICTED
        ii = find(trainsplit~=i);
        XX_pca = XX_pcaall(ii,:);
        [~,~,thislabels] = unique(trainlabel(ii));
    end
    iitest = find(splitid==i);
    iitrain = find(splitid~=i);
    
    for k = 1:length(lambdarange),
        thislambda = lambdarange(k);
        params.transfer_model.lambda = thislambda;
        % weight factor when combining current model and new model
        params.transfer_model.w = params.transfer_model.lambda/(1+params.transfer_model.lambda);
        % using the train folds to get new estimation for the joint-bayesian model
        if ~exist('cBJmodel','var') || size(cBJmodel,2)<i || size(cBJmodel,1)<k || isempty(cBJmodel{k,i})
            if updateInDomainJbModel
                jbFilePath = fullfile(recognitionResDir, sprintf('jb_lambda%0.2f_%d.mat', thislambda, i));
                if exist(jbFilePath, 'file')
                    load(jbFilePath, 'x');
                    cBJmodel{k,i} = x;
                else                    
                    if ~RESTRICTED
                        cBJmodel{k,i} = ...
                            get_EMLike_model_clean(XX_pca',thislabels,params);
                    else
                        posi = iitrain(find(ylong(iitrain)>0));
                        thisXX_pca = [XX1proj(posi,:);XX2proj(posi,:)];
                        thislabelsrest = [1:length(posi), 1:length(posi)];
                        cBJmodel{k,i} = ...
                            get_EMLike_model_clean(thisXX_pca',thislabelsrest,params);
                    end
                    x = cBJmodel{k,i};
                    save(jbFilePath, 'x');
                end
            else
                % use the existing model
                cBJmodel{k,i} = INDOJBMODEL;
            end
            
            scsjbtl(i,:,k) = ...
                classify_bayes_joint_form_clean(XX1proj',XX2proj',cBJmodel{k,i})';
            
            trainData = double(scsjbtl(i,iitrain,k));
            trainLabels = ylong(iitrain); % TODO: originally it was y - is it the same ??
            testData = double(scsjbtl(i,iitest,k));
            testLabels = ylong(iitest);
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
                    RRjbtl(k,i) = mean(sign(testRes) == testLabels')
                case 2
                    [ry,crwjbtl{i,k}] = CLSliblinearC(testData, MODEL);
                    %[~ ,crwjbtltrain{i,k}] = CLSliblinearC(double(scsjbtl(i,iitrain,k)), MODEL);
                    RRjbtl(k,i) = mean(ry == testLabels)
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
                    RRjbtl(k,i) = mean(matchDetections == testLabels')
            end
        end
    end
end
RRjbtl
[mean(RRjbtl(1,:)) std(RRjbtl(1,:))]