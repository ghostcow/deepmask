imSize = [152, 152];        
datasetFilePathFormat = '../data_files/aligned/cfw_flat';
trainSize = 17920 + 16640;
testSize = 14848;

train = single(zeros(trainSize, 3, imSize(1), imSize(2)));
trainLabels = uint16(zeros(trainSize, 1));
test = single(zeros(14848, 3, imSize(1), imSize(2)));
testLabels = uint16(zeros(14848, 1));

S = load([datasetFilePathFormat '_train_1.mat']);
nImages = length(S.labels);
train(1:nImages, :, :, :) = S.data;
trainLabels(1:nImages) = S.labels;

S = load([datasetFilePathFormat '_train_2.mat']);
nImages2 = length(S.labels);
train((nImages+1):trainSize, :, :, :) = S.data;
trainLabels((nImages+1):trainSize) = S.labels;

S = load([datasetFilePathFormat '_test_1.mat']);
nImages = length(S.labels);
test(1:nImages, :, :, :) = S.data;
testLabels(1:nImages) = S.labels;

save([datasetFilePathFormat '.mat'], 'train', 'trainLabels', 'test', 'testLabels', '-v7.3');