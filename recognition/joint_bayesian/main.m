% QUESTIONS :
% 1. what are lfwX,trainsplit, trainlabel ? (relevant only when RESTRICTED = false)
%
% guess -
% lfwX is the people.txt data (unrestricted) with the appropriate
% trainsplit/trainlabel indicating split id and labels
clear all; close all; clc;
addpath('..');
addpath('../../../liblinear-1.95/matlab');

lfwDir = 'D:\_Dev\Datasets\Face Recognition\LFW';
pairsFilePath = fullfile(lfwDir, '/view2/pairs.txt');
peopleFilePath = fullfile(lfwDir, '/view2/people.txt');

%% Loading pairs data (10 splits bundled together)
LoadPairsData;
numFolds = 10;
numPairsPerFold = 300;
% lfw configuration : restricted/unrestrcited
RESTRICTED = 1; % original value = 0
% should update the in-domain trained jb model, using the new domain data
updateInDomainJbModel = false; % original value = true
svmParams = struct('type', 1, 'C', 1); % original values : type=3, C=0.05
% finalTestMethod : 
% 1 = liblinear train & predict, 2 = liblinear train & our predict, 3 = choosing best threshold based on training
finalTestMethod = 3; 

% convert lfw names to numbers (directories indices)
figDirs = dir(fullfile(lfwDir, 'lfw'));
figDirs = figDirs(3:end);
figDirs = {figDirs.name};
mapNameToNum = containers.Map;
for iDir = 1:length(figDirs)
    mapNameToNum(figDirs{iDir}) = iDir;
end

if ~RESTRICTED
    % TODO: should load
    % lfwx       - Nxd feature vectors for all face images (load from mat file)
    % trainlabel - person label for each face image
    % trainsplit - split id for each face image
    peopleMetadata = GetPeopleData(peopleFilePath);
    
    %     lfwx = load('lfw_people.mat');
    %     lfwx = lfwx.x;
    nImages = 20000; %size(lfwx, 2);
    trainlabel = zeros(1, nImages);
    trainsplit = zeros(1, nImages);
    for iFold = 1:numFolds
        for iPerson = 1:length(peopleMetadata{iFold})
            trainlabel(peopleMetadata{iFold}(iPerson).imageIndices) = ...
                mapNameToNum(peopleMetadata{iFold}(iPerson).name);
            trainsplit(peopleMetadata{iFold}(iPerson).imageIndices) = iFold;
        end
    end
end

% TEMP - loading lfw relevant data for training pca & joint-baysian learning
[namesLeft, namesRight, imgNumLeft, imgNumRight] = ...
    ParsePairsFile(pairsFilePath);
idsLeft = cellfun(@(x)(mapNameToNum(x)), namesLeft);
idsRight = cellfun(@(x)(mapNameToNum(x)), namesRight);

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
    ylong = labels;
    splitid = zeros(1, size(XX1, 2));
    for iFold = 1:numFolds
        startIndex = 1 + (iFold-1)*2*numPairsPerFold;
        splitid(startIndex:(startIndex+2*numPairsPerFold-1)) = iFold;
    end
    
    % TEMP - pca & joint-baysian learning using the LFW data (eventually it
    % will be done with other external data)
    XSOURCEDOMAIN = [XX1'; XX2'];
    ysource = [idsLeft, idsRight];
end

%% PCA & joint-bayesian model learning, using extranal data
outputDim = 299; %first parameter I tried
load('after_pca.mat');
if ~exist('INDOPCAModel','var')
    [XX_pca,INDOPCAModel] = WPCA_v2(XSOURCEDOMAIN,[],outputDim,[]);
end

load('EM.mat');
if ~exist('INDOJBMODEL','var')
    INDOJBMODEL = get_EMLike_model_clean(XX_pca',ysource',[]);
end

params.initial_guess.Se = INDOJBMODEL.Se;
params.initial_guess.Sm = INDOJBMODEL.Sm;
params.transfer_model.Se = INDOJBMODEL.Se;
params.transfer_model.Sm = INDOJBMODEL.Sm;
params.transfer_model.lambda = 0.5;
params.transfer_model.w = params.transfer_model.lambda/(1+params.transfer_model.lambda);

%% pca projection over the test data
XX1proj = WPCA_v2(XX1',INDOPCAModel,outputDim,[]);
XX2proj = WPCA_v2(XX2',INDOPCAModel,outputDim,[]);
if ~RESTRICTED
    XX_pcaall = WPCA_v2(lfwX,INDOPCAModel,outputDim,[]);
end

cPCAModel = cell(1);
cBJmodel = cell(1);

lambdarange = [1.35]; %not the first parameter I tried

for i = [5,setdiff(1:10,5)]
    disp(i);
    if ~RESTRICTED
        XX_pca = XX_pcaall(ii,:);
        ii = find(trainsplit~=i);
        [~,~,thislabels] = unique(trainlabel(ii));
    else
        iitest = find(splitid==i);
        iitrain = find(splitid~=i);
    end
    
    for k = 1:length(lambdarange),
        thislambda = lambdarange(k);
        params.transfer_model.lambda = thislambda;
        % weight factor when combining current model and new model
        params.transfer_model.w = params.transfer_model.lambda/(1+params.transfer_model.lambda);
        % using the train folds to get new estimation for the joint-bayesian model
        if ~exist('cBJmodel','var') || size(cBJmodel,2)<i || size(cBJmodel,1)<k || isempty(cBJmodel{k,i})
            if updateInDomainJbModel
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
            else
                % use the existing model
                cBJmodel{k,i} = INDOJBMODEL;
            end
            
            scsjbtl(i,:,k) = ...
                classify_bayes_joint_form_clean(XX1proj',XX2proj',cBJmodel{k,i})';
            
            trainData = double(scsjbtl(i,iitrain,k));
            trainLabels = ylong(iitrain); % TODO: originally it was y - is it the same ??
            MODEL = CLSliblinear(sparse(trainData), trainLabels, svmParams);
            testData = double(scsjbtl(i,iitest,k));
            testLabels = ylong(iitest);
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

[mean(RRjbtl(1,:)) std(RRjbtl(1,:))]