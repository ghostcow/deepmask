function [targetPeopleX, targetPeopleY, targetPeopleSplitId] = LoadLfwPeople(lfwpeopleResFilePath, lfwPeopleImagesFilePath)
% load features for LFW people images (as decalred in people.txt)
% INPUT
%   lfwpeopleResFilePath - path to features file or a pattern of more than 1 file (for
%       example /features_*.mat), in this case the features are concatenated
%   lfwPeopleImagesFilePath - path to mat file with the people labels and
%       split ids
% OUTPUT : 
%   targetPeopleX - features for all LFW people
%   targetPeopleY - identity label for each feature
%   targetPeopleSplitId - split id (1-10) for each feature

% load labels & split ids
S = load(lfwPeopleImagesFilePath, 'labels', 'splitId');
nImages = length(S.labels);
targetPeopleY = S.labels;
targetPeopleSplitId = S.splitId;

% load features
lfwpeopleResFiles = dir(lfwpeopleResFilePath);
resFileRoot = fileparts(lfwpeopleResFilePath);

nFiles = length(lfwpeopleResFiles);
% final 2D array with pairs feature, each with dimensions [featureDim x nPairs]
targetPeopleX = []; 
for iFile = 1:nFiles
    S = load(fullfile(resFileRoot, lfwpeopleResFiles(iFile).name));
    if isempty(targetPeopleX)
        subFeatureDim = size(S.x, 1);
        featureDim = nFiles*subFeatureDim;
        targetPeopleX = zeros(featureDim, nImages);
    end
    subFeatureIndex = 1 + (iFile - 1)*subFeatureDim;
    targetPeopleX(subFeatureIndex:(subFeatureIndex + subFeatureDim - 1), :) = S.x(:, 1:end);
end
    
end