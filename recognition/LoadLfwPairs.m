function [targetX1, targetX2, targetY, targetSplitId] = LoadLfwPairs(resFilePath, maxNumNets)
% load features for LFW pairs images (as decalred in pairs.txt)
% INPUT
%   resFilePath - path to features file or a pattern of more than 1 file (for
%       example /features_*.mat), in this case the features are concatenated
% OUTPUT : 
%   targetX1, targetX2 - feature pairs
%   targetY - label for each pair (1/-1)
%   targetSplitId - split id (1-10) for each pair

if ~exist('maxNumNets', 'var')
    maxNumNets = inf;
end

% constants
numFolds = 10;
numPairsPerFold = 300; % half of #pairs in each fold (positive/negative pairs)
nPairs = numFolds*numPairsPerFold*2;

% load by pairs (based on pairs.txt)
lfwpairsResFiles = dir(resFilePath);
resFileRoot = fileparts(resFilePath);
nFiles = length(lfwpairsResFiles);
nFiles = min(nFiles, maxNumNets);

% final 2D array with pairs feature, each with dimensions [featureDim x nPairs]
targetX1 = []; 
targetX2 = [];
for iFile = 1:nFiles
    S = load(fullfile(resFileRoot, lfwpairsResFiles(iFile).name));
    if isempty(targetX1)
        subFeatureDim = size(S.x, 1);
        featureDim = nFiles*subFeatureDim;
        targetX1 = zeros(featureDim, nPairs);
        targetX2 = zeros(featureDim, nPairs);
    end
    subFeatureIndex = 1 + (iFile - 1)*subFeatureDim;
    targetX1(subFeatureIndex:(subFeatureIndex + subFeatureDim - 1), :) = S.x(:, 1:2:end);
    targetX2(subFeatureIndex:(subFeatureIndex + subFeatureDim - 1), :) = S.x(:, 2:2:end);
end

targetY = zeros(nPairs, 1);
targetSplitId = zeros(size(targetX1, 2), 1);
for iFold = 1:numFolds
    startIndex = 1 + (iFold-1)*2*numPairsPerFold;
    targetSplitId(startIndex:(startIndex+2*numPairsPerFold-1)) = iFold;
    targetY(startIndex:(startIndex+numPairsPerFold-1)) = 1; % positive pairs
    targetY((startIndex+numPairsPerFold):(startIndex+2*numPairsPerFold-1)) = -1; % negative pairs
end

end