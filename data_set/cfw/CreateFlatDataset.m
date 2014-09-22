% create subsets where all person have 30-80 training images

outputFilePath = 'cfw_flat_1001_1440.mat';
trainPerc = 0.7;
imSize = [152,152];
mainDir = '/media/data/datasets/CFW/filtered_faces';
figDirs = dir(mainDir);
figDirs = figDirs(3:end);
    
trainingSamples = [30 80];
maxTestSamples = ceil(trainingSamples(2)*trainPerc);

%% Initialize new dataset
train = zeros(0, 3, imSize(1), imSize(2));
trainLabels = zeros(0, 1);
test = zeros(0, 3, imSize(1), imSize(2));
testLabels = zeros(0, 1);

%%
subsetsNames = {'cfw_1_574.mat', 'cfw_575_1000.mat', 'cfw_1001_1440.mat'};

% for iSubset = 1:length(subsetsNames)
    iSubset = 3;
    S = load(subsetsNames{iSubset});
    personsLabels = unique(S.trainLabels);
    fprintf('%s : %d persons\n', subsetsNames{iSubset}, length(personsLabels));
    for iPerson = personsLabels'
        indicesTrain = find(S.trainLabels == iPerson);
        indicesTest = find(S.testLabels == iPerson);
        numTrainingSamples = length(indicesTrain);

        if (numTrainingSamples > trainingSamples(1))
            fprintf('%d - %d\n', iPerson, numTrainingSamples);
            
            % re-arrange training & test images into one stack
            currPersonImages = cat(1, S.train(indicesTrain, :, :, :), S.test(indicesTest, :, :, :));

            if (numTrainingSamples > trainingSamples(2))
                % too many samples - choose images with best resolution
                imagesDir = fullfile(mainDir, figDirs(iPerson).name);
                images = dir(fullfile(imagesDir, '*.png'));
                nImages = length(images);
                imageAreas = zeros(1, nImages);
                for iImage = 1:nImages
                    imagePath = fullfile(imagesDir, images(iImage).name);
                    imageInfo = imfinfo(imagePath);
                    imageAreas(iImage) = imageInfo.Height * imageInfo.Width;
                end
                [~, bigAreaImagesIndice] = sort(imageAreas, 'descend');
                
                numImagesToTake = min(trainingSamples(2) + maxTestSamples, nImages);
                
                % shuffle best resolution images indices (to mix images between train & test)
                randIndicesToTake = randperm(numImagesToTake);
                randIndicesToTake = bigAreaImagesIndice(randIndicesToTake);
                
                indicesTrain = randIndicesToTake(1:trainingSamples(2));
                indicesTest = randIndicesToTake((trainingSamples(2)+1):end);
            else
                indicesTrain = 1:length(indicesTrain);
                indicesTest = 1:length(indicesTest);
            end

            % copy into global arrays
            train = cat(1, train, currPersonImages(indicesTrain, :, :, :));
            trainLabels = cat(1, trainLabels, iPerson*ones(length(indicesTrain), 1));
            test = cat(1, test, currPersonImages(indicesTest, :, :, :));
            testLabels = cat(1, testLabels, iPerson*ones(length(indicesTest), 1));
        end
    end
% end

save(outputFilePath, 'train', 'trainLabels', 'test', 'testLabels', '-v7.3');