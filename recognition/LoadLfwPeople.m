function [targetPeopleX, targetPeopleY, targetPeopleSplitId, imPaths] = ...
    LoadLfwPeople(lfwpeopleResFilePath, lfwPeopleImagesFilePath, netIndices)
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

if ~exist('netIndices', 'var')
    netIndices = [];
end

% load labels & split ids
S = load(lfwPeopleImagesFilePath);
nImages = length(S.labels);
targetPeopleY = S.labels;
targetPeopleSplitId = S.splitId;
imPaths = S.imPaths;

% load features
lfwpeopleResFiles = dir(lfwpeopleResFilePath);
resFileRoot = fileparts(lfwpeopleResFilePath);

nFiles = length(lfwpeopleResFiles);
if isempty(netIndices)
    netIndices = 1:nFiles;
end
nFiles = length(netIndices);

% final 2D array with pairs feature, each with dimensions [featureDim x nPairs]
targetPeopleX = []; 
iFile = 1;
for netIndex = netIndices
    S = load(fullfile(resFileRoot, lfwpeopleResFiles(netIndex).name));
    if isempty(targetPeopleX)
        subFeatureDim = size(S.x, 1);
        featureDim = nFiles*subFeatureDim;
        targetPeopleX = zeros(featureDim, nImages);
    end
    subFeatureIndex = 1 + (iFile - 1)*subFeatureDim;
    targetPeopleX(subFeatureIndex:(subFeatureIndex + subFeatureDim - 1), :) = S.x(:, 1:end);
    iFile = iFile + 1;
end

% images with lable=0 are invalid (face detections has been failed)
validImagesIndices = find(targetPeopleY > 0);
targetPeopleX = targetPeopleX(:, validImagesIndices);
targetPeopleY = targetPeopleY(validImagesIndices);
targetPeopleSplitId = targetPeopleSplitId(validImagesIndices);
imPaths = imPaths(validImagesIndices);

end