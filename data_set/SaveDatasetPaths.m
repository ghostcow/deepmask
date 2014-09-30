% This script save the dataset image paths into csv file

%% change paths here
mainDir = '/media/data/datasets/CFW/filtered_aligned';
outputFilePath = 'cfw/images';

% output txt files
outputFilePathTrain = [outputFilePath '_train.txt'];
outputFilePathTest = [outputFilePath '_test.txt'];

%% 0.7 of the images are used for training, the rest for test
trainPerc = 0.7;

% optional - defining minimum & maximum samples per person
trainingSamples = [30 80]; % use [0 inf] to take all images
maxTestSamples = ceil(trainingSamples(2)*trainPerc);

% start iterating
figDirs = dir(mainDir);
figDirs = figDirs(3:end);
nPersons = length(figDirs);
fidTrain = fopen(outputFilePathTrain, 'w');
fidTest = fopen(outputFilePathTest, 'w');
for iPerson = 1:nPersons
    fprintf('%d - %s\n', iPerson, figDirs(iPerson).name);
    imagesDir = fullfile(mainDir, figDirs(iPerson).name);
    images = dir(fullfile(imagesDir, '*.jpg'));
    
    nImages = length(images);
    nTrainingSamples = round(nImages*trainPerc);
    if (nTrainingSamples > trainingSamples(1))
        if (nTrainingSamples > trainingSamples(2))        
            nImagesToTake = min(trainingSamples(2) + maxTestSamples, nImages);
            nTrainingSamples = round(nImagesToTake*trainPerc);
            imageIndices = randperm(nImages);
            
            % too many samples - choose images with best resolution
            % (this is can't done now because now all images are of size 152 x
            % 152)
%             imageAreas = zeros(1, nImages);
%             for iImage = 1:nImages
%                 imagePath = fullfile(imagesDir, images(iImage).name);
%                 imageInfo = imfinfo(imagePath);
%                 imageAreas(iImage) = imageInfo.Height * imageInfo.Width;
%             end
%             [~, bigAreaImagesIndice] = sort(imageAreas, 'descend');
%             imageIndices = bigAreaImagesIndice(randperm(nImagesToTake));
        else
            nImagesToTake = nImages;
            imageIndices = randperm(nImages);
        end
        
        % write image paths to files
        for iImage = 1:nImagesToTake
            imagePath = fullfile(imagesDir, images(imageIndices(iImage)).name);
            if (iImage <= nTrainingSamples)
                fprintf(fidTrain, '%s,%d\n', imagePath, iPerson);
            else
                fprintf(fidTest, '%s,%d\n', imagePath, iPerson);
            end
        end
    end
end  
fclose(fidTrain);
fclose(fidTest);
   