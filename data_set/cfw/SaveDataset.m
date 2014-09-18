% outputFilePath = 'cfw_small.mat';
% nPersons = 200

% The full dataset is saved in 3 chunks to avoid OUT OF MEMORY error
outputFilePath = 'cfw_575_1000.mat';

mainDir = '/media/data/datasets/CFW/filtered_faces';
figDirs = dir(mainDir);
figDirs = figDirs(3:end);
imSize = [152,152];

%% 0.7 of the images are used for training, the rest for test
trainPerc = 0.7;

numImagesMax = 50000;
numImagesMaxTrain = round(numImagesMax*trainPerc);
numImagesMaxTest = numImagesMax - numImagesMaxTrain;

train = zeros(numImagesMaxTrain, 3, imSize(1), imSize(2));
trainLabels = zeros(numImagesMaxTrain, 1);
test = zeros(numImagesMaxTest, 3, imSize(1), imSize(2));
testLabels = zeros(numImagesMaxTest, 1);
iImageTrain = 1;
iImageTest = 1;
for iFigure = 575:1000 % 1:nPersons
    disp(iFigure);
    currDir = fullfile(mainDir, figDirs(iFigure).name);
    images = dir(fullfile(currDir, '*.png'));
    nImages = length(images);
    nTraining = round(nImages*trainPerc);
    isTrain = [ones(nTraining, 1); zeros(nImages - nTraining, 1)];
    
    currFigTrain = zeros(0, 3, imSize(1), imSize(2));
    currFigTest = zeros(0, 3, imSize(1), imSize(2));
    
    for iImage = 1:nImages
        [im, map] = imread(fullfile(currDir, images(iImage).name));
        if (size(im, 3) == 1)
            if ~isempty(map)
                im = ind2rgb(im, map);
            else
                error('grayscale image');
            end
        end
        
        im = im2double(im);
        im = imresize(im, imSize);
        % convert shape from 152x152x3 3x152x152
        im = shiftdim(im, 2); 
        
        if isTrain(iImage)
            train(iImageTrain,:,:,:) = im;
            trainLabels(iImageTrain) = iFigure;
            iImageTrain = iImageTrain + 1;
            %currFigTrain(end+1,:,:,:) = im;
        else
            test(iImageTest,:,:,:) = im;
            testLabels(iImageTest) = iFigure;
            iImageTest = iImageTest + 1;
            %currFigTest(end+1,:,:,:) = im;
        end
    end
    
    % copy into global arrays
    %train = cat(1, train, currFigTrain);
    %trainLabels = cat(1, trainLabels, iFigure*ones(size(currFigTrain,1), 1));
    %test = cat(1, test, currFigTest);
    %testLabels = cat(1, testLabels, iFigure*ones(size(currFigTest,1), 1));
end
% cut unused cells
train(iImageTrain:numImagesMaxTrain,:,:,:) = [];
trainLabels(iImageTrain:numImagesMaxTrain) = [];
test(iImageTest:numImagesMaxTest,:,:,:) = [];
testLabels(iImageTest:numImagesMaxTest) = [];

save(outputFilePath, 'train', 'trainLabels', 'test', 'testLabels', '-v7.3');