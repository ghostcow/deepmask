% This script read image paths from csv files and generate mat files containing
% the images (randomized)

% should use different mat file to train/test
useDifferentFiles = false;
randomPermutation = false;
maxImagesPerFile = 5000; % relevant only if useDifferentFiles=true

%% change values here
name = 'cfw_pubfig_sufr';
if strcmp(name, 'cfw_pubfig_sufr')
    inputFilePath = '../data/deepId_full/CFW_PubFig_SUFR/images';
    outputFilePathFormat = '../data_files/deepId_full/CFW_PubFig_SUFR/CFW_PubFig_SUFR';
    useDifferentFiles = true;
    scaleFactor = 0.5;
end

% input txt files
inputFilePathTrain = [inputFilePath '_train.txt'];
inputFilePathTest = [inputFilePath '_test.txt'];
setTypes = {'train', 'test'};
inputFilePaths = {inputFilePathTrain, inputFilePathTest};

inputFilePathVer = [inputFilePath '_verification.txt'];
if exist(inputFilePathVer, 'file')
    setTypes{end+1} = 'verification';
    inputFilePaths{end+1} = inputFilePathVer;
    
    % verification data is always saved as one file
    verification = [];
    verificationLabels = [];
end

if ~useDifferentFiles
    train = [];
    trainLabels = [];
    test = [];
    testLabels = [];
end

%% Load all images and save into mat files
for iSet = 1:length(setTypes)
    setType = setTypes{iSet};
    fid = fopen(inputFilePaths{iSet});
    C = textscan(fid, '%s %d','delimiter', ',');
    fclose(fid);
    nImages = length(C{1});
    if randomPermutation
        imageIndices = randperm(nImages);
    else
        imageIndices = 1:nImages;
    end
    
    nFiles = ceil(nImages/maxImagesPerFile);
    iStart = 1;
    for iFile = 1:nFiles
        outputFilePath = [outputFilePathFormat '_' setType '_' num2str(iFile) '.mat'];
        iEnd = min(iStart + maxImagesPerFile - 1, nImages);
        chunkSize = iEnd - iStart + 1;

        data = []; 
        labels = uint16(zeros(chunkSize, 1));
        jImage = 1;
        for iImage = iStart:iEnd
            imagePath = C{1}{imageIndices(iImage)};
            imageLabel = C{2}(imageIndices(iImage));

            [im, map] = imread(imagePath);
            if ~isempty(map)
                im = ind2rgb(im, map);
            end
            if (size(im, 3) == 1)
                im = cat(3, im, im, im);
            end

            im = im2single(im);
            if (scaleFactor ~= 1)
                im = imresize(im, scaleFactor);
            end
            
            % convert shape from 152x152x3 3x152x152
            im = shiftdim(im, 2); 
            
            if isempty(data)
                data = single(zeros(chunkSize, 3, size(im, 2), size(im, 3)));
            end
            data(jImage,:,:,:) = im;
            labels(jImage) = imageLabel;
            jImage = jImage + 1;
        end
        if (strcmp(setType, 'verification'))
            verification = cat(1, verification, data);
            verificationLabels = cat(1, verificationLabels, labels);
            if (iFile == nFiles)
                data = verification;
                labels = verificationLabels;
                save([outputFilePathFormat '_' setType '.mat'], 'data', 'labels', '-v7.3');
            end
        else
            if (useDifferentFiles)
                save(outputFilePath, 'data', 'labels', '-v7.3');
            else
                if strcmp(setType, 'train')
                    train = cat(1, train, data);
                    trainLabels = cat(1, trainLabels, labels);
                elseif strcmp(setType, 'test')
                    test = cat(1, test, data);
                    testLabels = cat(1, testLabels, labels);
                    if (iFile == nFiles)
                        save([outputFilePathFormat '.mat'], 'train', 'trainLabels', 'test', 'testLabels', '-v7.3');
                        clear train trainLabels test testLabels
                    end
                end
            end
        end
        iStart = iEnd + 1;
    end
end
if ~useDifferentFiles
    
end