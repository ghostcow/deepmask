clear variables; clc;

%% constants
disyType = 'jb'; % jb / L2 / cosine
useGpu = true;

resultType = 5;
resultName = '.';
netIndices = 1:10;

% resultType = 4;
% resultName = 'patch_1_30';
% netIndices = []; %1:10;

%% Loading the PCA & JB models
lfwPeopleImagesFilePath = '../data_files/deepId_full/LFW/people.mat';
[resDir, lfwpairsResFileName, lfwpeopleResFileName, verificationResFileName, verificationImagesFilePath] = ...
    GetResultFilePaths(resultType);
recognitionResDir = fullfile(resDir, 'recognition', resultName);
disp(recognitionResDir);

% Load PCA & JB models
pcaFilePath = fullfile(recognitionResDir, 'pca_verification.mat');
jbFilePath = fullfile(recognitionResDir, 'jb_verification.mat');

load(pcaFilePath, 'sourceXpca', 'sourcePcaModel');
load(jbFilePath, 'sourceJbModel');

outputDim = size(sourcePcaModel.full_proj, 2);
jbParams = struct;
jbParams.initial_guess.Se = sourceJbModel.Se;
jbParams.initial_guess.Sm = sourceJbModel.Sm;
jbParams.transfer_model.Se = sourceJbModel.Se;
jbParams.transfer_model.Sm = sourceJbModel.Sm;
jbParams.transfer_model.lambda = 0.5;
jbParams.transfer_model.w = jbParams.transfer_model.lambda/(1+jbParams.transfer_model.lambda);

%% Load LFW gallery & probe
[galleryX, galleryY, probeX, probeY] = ...
    GetLfwIdentificationData(fullfile(resDir, lfwpeopleResFileName), lfwPeopleImagesFilePath, netIndices);
galleryX = single(wpca(galleryX', sourcePcaModel)');
probeX = single(wpca(probeX', sourcePcaModel)');
gallerySize = size(galleryX, 2);
probeSize = size(probeX, 2);

%% serach all probe faces in gallery
A = zeros(probeSize, gallerySize);
avgDistTime = 0;
if (strcmp(disyType, 'jb'))
    if useGpu
        probeX = gpuArray(probeX);
        sourceJbModel.A = gpuArray(sourceJbModel.A);
        sourceJbModel.G = gpuArray(sourceJbModel.G);
    end
    % th = -14.1; % average of 10 best th for lfw 10 folds
    for iProbe = 1:probeSize
        queryFeature = probeX(:, iProbe);
        tic;
        jbScores = jointBayesianC(queryFeature, galleryX, sourceJbModel);
        avgDistTime = avgDistTime + toc;
        if useGpu
            jbScores = gather(jbScores);
        end
        A(iProbe, :) = jbScores;
    end
    avgDistTime = avgDistTime / probeSize;
    fprintf('average distance computation time = %f [sec]\n', avgDistTime);
elseif (strcmp(disyType,'L2'))
    A = pdist2(probeX', galleryX', 'cosine');
elseif (strcmp(disyType,'cosine'))
    A = pdist2(probeX', galleryX', 'cosine');
end

% topk accuracy
[sortedDists, nnIndices] = sort(A, 2, 'descend');
ks = [1,5,10,20,30,50,100];
fprintf('#people(=faces) in gallery = %d\n', gallerySize);
fprintf('#faces in probe = %d\n', probeSize);
for iK = 1:length(ks)
    k = ks(iK);    
    nSuccess = 0;
    for iProbe = 1:probeSize   
        imageNN = nnIndices(iProbe, :);
        temp = ismember(probeY(iProbe), galleryY(imageNN(1:k)));
        nSuccess = nSuccess + double(temp);
    end
    accuracy = 100*(nSuccess / probeSize);
    fprintf('top %d accuracy = %f\n', k, accuracy);
end