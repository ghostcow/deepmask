% This script save the dataset image paths into csv file, where each line
% format is : [image path],[label]

%% change paths here
mainDirs = {'/media/data/datasets/CFW/aligned_deepid'}; %{'/media/data/datasets/CFW/filtered_aligned_network/byFigure', '/media/data/datasets/CFW/filtered_aligned_small'};
outputFilePath = '../data/deepId/temp_images'; %'../data/CFW_clean/images';

% output txt files
outputFilePathTrain = [outputFilePath '_train.txt'];
outputFilePathTest = [outputFilePath '_test.txt'];

%% 0.7 of the images are used for training, the rest for test
trainPerc = 0.75;

% optional - defining minimum & maximum samples per person
trainingSamples = [0 10]; % use [0 inf] to take all images
maxTestSamples = ceil(trainingSamples(2)*(1-trainPerc));

nPersonsTot = 0;
nImagesTot = 0;
fidTrain = fopen(outputFilePathTrain, 'w');
fidTest = fopen(outputFilePathTest, 'w');
iLabel = 1;
for iDir = 1:length(mainDirs)
    mainDir = mainDirs{iDir};
    % start iterating
    figDirs = dir(mainDir);
    figDirs = figDirs(3:end);
    nPersons = length(figDirs);

    % contiguos labeling of the persons chosen for the dataset
    nPersonsTot = nPersonsTot + nPersons;
    for iPerson = 1:nPersons
        fprintf('%d - %s\n', iPerson, figDirs(iPerson).name);
        imagesDir = fullfile(mainDir, figDirs(iPerson).name);
        images = dir(fullfile(imagesDir, '*.jpg'));

        nImages = length(images);
        nImagesTot = nImagesTot + nImages;
        nTrainingSamples = round(nImages*trainPerc);
        if (nTrainingSamples > trainingSamples(1))
            if (nTrainingSamples > trainingSamples(2))        
                nImagesToTake = min(trainingSamples(2) + maxTestSamples, nImages);
                nTrainingSamples = round(nImagesToTake*trainPerc);
                imageIndices = randperm(nImages);
            else
                nImagesToTake = nImages;
                imageIndices = randperm(nImages);
            end
            disp(nImagesToTake);

            % write image paths to files
            for iImage = 1:nImagesToTake
                imagePath = fullfile(imagesDir, images(imageIndices(iImage)).name);
                if (iImage <= nTrainingSamples)
                    fprintf(fidTrain, '%s,%d\n', imagePath, iLabel);
                else
                    fprintf(fidTest, '%s,%d\n', imagePath, iLabel);
                end
            end
            iLabel = iLabel + 1;
        end
    end  
end
fclose(fidTrain);
fclose(fidTest);
   