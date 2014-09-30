% This script read image paths from files and generate mat files containing
% the images

% 
maxImagesPerFile = 18000;
batchSize = 128;
imSize = [152 152];

%% change paths here
outputFilePathFormat = '../data_files/aligned/cfw_flat';

% input txt files
inputFilePath = 'cfw/images';
inputFilePathTrain = [inputFilePath '_train.txt'];
inputFilePathTest = [inputFilePath '_test.txt'];

%% 
setTypes = {'train', 'test'};
inputFilePaths = {inputFilePathTrain, inputFilePathTest};

for iSet = 2 %1:2
    setType = setTypes{iSet};
    fid = fopen(inputFilePaths{iSet});
    C = textscan(fid, '%s %d','delimiter', ',');
    fclose(fid);
    nImages = length(C{1});

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
            imagePath = C{1}{iImage};
            imageLabel = C{2}(iImage);

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
        save(outputFilePath, 'data', 'labels', '-v7.3');
        iStart = iEnd + 1;
    end
end