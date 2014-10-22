clear all;
close all;


%% Define all paths here
mainDir = '/media/data/datasets/CFW';
allImagesDir = fullfile(mainDir, 'images');
alignedImagesDir = fullfile(mainDir, 'filtered_aligned_affine');

%%
detectionsFileName = 'detections_miley_cyrus.txt';
fid = fopen(detectionsFileName);
A = textscan(fid, '%s %s %s', 'Delimiter', ',');
nDetections = length(A{1});
isValidDetections = ones(1, nDetections);
for iDetection = 1:nDetections
    imPath = A{1}{iDetection};
    alignedImPath = A{2}{iDetection};
    
    if ~exist(alignedImPath, 'file')
        isValidDetections(iDetection) = false;
    end
    % detectionsStr = A{3}{iDetection};
end
fclose(fid);

fid = fopen(detectionsFileName, 'w');
validDetectionsIndices = find(isValidDetections);
for iDetection = validDetectionsIndices
    fprintf(fid, '%s,%s,%s\n', A{1}{iDetection}, A{2}{iDetection}, A{3}{iDetection});
end
fclose(fid);