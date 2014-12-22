% data
dataDir = '../../data_files/deepId_full/CASIA/';
testFiles = dir(fullfile(dataDir, 'CASIA_test_*.mat'));
targetFile = fullfile(dataDir, 'CASIA_verification.mat');

% data
data = [];
labels = [];
data = single(data);
labels = uint16(labels);
for iFile = 1:length(testFiles)
    S = load(fullfile(dataDir, testFiles(iFile).name));
    data = cat(1, data, S.data);
    labels = cat(1, labels, S.labels);
end

save(targetFile, 'data', 'labels', '-v7.3');