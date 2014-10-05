datasetFilePathFormat = '../data_files/CFW_small/cfw_small';

imSize = [152, 152];        
trainSize = 0;
testSize = 0;

train = single(zeros(trainSize, 3, imSize(1), imSize(2)));
trainLabels = uint16(zeros(trainSize, 1));
test = single(zeros(testSize, 3, imSize(1), imSize(2)));
testLabels = uint16(zeros(testSize, 1));

S = load([datasetFilePathFormat '_train_1.mat']);
nImages = length(S.labels);
train(1:nImages, :, :, :) = S.data;
trainLabels(1:nImages) = S.labels;

% S = load([datasetFilePathFormat '_train_2.mat']);
% nImages2 = length(S.labels);
% train((nImages+1):trainSize, :, :, :) = S.data;
% trainLabels((nImages+1):trainSize) = S.labels;

S = load([datasetFilePathFormat '_test_1.mat']);
nImages = length(S.labels);
test(1:nImages, :, :, :) = S.data;
testLabels(1:nImages) = S.labels;

save([datasetFilePathFormat '.mat'], 'train', 'trainLabels', 'test', 'testLabels', '-v7.3');