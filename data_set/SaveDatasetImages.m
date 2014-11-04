% This script read image paths from csv files and generate mat files containing
% the images (randomized)

% 
useDifferentFiles = false;
batchSize = 128;
imSize = [152 152];
maxImagesPerFile = 40000; % relevant only if useDifferentFiles=true

%% change paths here
inputFilePath = '../data/deepId/CFW_small2/images';
outputFilePathFormat = '../data_files/deepId/CFW_small2/cfw_small2';

% input txt files
inputFilePathTrain = [inputFilePath '_train.txt'];
inputFilePathTest = [inputFilePath '_test.txt'];

%% 
setTypes = {'train', 'test'};
inputFilePaths = {inputFilePathTrain, inputFilePathTest};
if ~useDifferentFiles
    train = [];
    trainLabels = [];
    test = [];
    testLabels = [];
end

for iSet = 1:2
    setType = setTypes{iSet};
    fid = fopen(inputFilePaths{iSet});
    C = textscan(fid, '%s %d','delimiter', ',');
    fclose(fid);
    nImages = length(C{1});
    randIndices = randperm(nImages);
    
    nFiles = ceil(nImages/maxImagesPerFile);
    iStart = 1;
    for iFile = 1:nFiles
        outputFilePath = [outputFilePathFormat '_' setType '_' num2str(iFile) '.mat'];
        iEnd = min(iStart + maxImagesPerFile - 1, nImages);
        % make chunk size divided by batchSize
        chunkSize = iEnd - iStart + 1;
        iEnd = iEnd - mod(chunkSize, batchSize);
        chunkSize = iEnd - iStart + 1;

        data = single(zeros(chunkSize, 3, imSize(1), imSize(2)));
        labels = uint16(zeros(chunkSize, 1));
        jImage = 1;
        for iImage = iStart:iEnd
            imagePath = C{1}{randIndices(iImage)};
            imageLabel = C{2}(randIndices(iImage));

            [im, map] = imread(imagePath);
            if ~isempty(map)
                im = ind2rgb(im, map);
            end
            if (size(im, 3) == 1)
                im = cat(3, im, im, im);
            end

            im = im2single(im);
            % convert shape from 152x152x3 3x152x152
            im = shiftdim(im, 2); 

            data(jImage,:,:,:) = im;
            labels(jImage) = imageLabel;
            jImage = jImage + 1;
        end
        if useDifferentFiles
            save(outputFilePath, 'data', 'labels', '-v7.3');
        else
            if strcmp(setType, 'train')
                train = cat(1, train, data);
                trainLabels = cat(1, trainLabels, labels);
            elseif strcmp(setType, 'test')
                test = cat(1, test, data);
                testLabels = cat(1, testLabels, labels);    
            end
        end
        iStart = iEnd + 1;
    end
end
if ~useDifferentFiles
    save([outputFilePathFormat '.mat'], 'train', 'trainLabels', 'test', 'testLabels', '-v7.3');
end