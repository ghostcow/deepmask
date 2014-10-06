alignedImagesDir = '/media/data/datasets/CFW/filtered_aligned';
filteredAlignedImagesDir = '/media/data/datasets/CFW/filtered_aligned_small';
outputFilePath = 'data/images';
% output txt files
outputFilePathTrain = [outputFilePath '_train.txt'];
outputFilePathTest = [outputFilePath '_test.txt'];

%% 0.7 of the images are used for training, the rest for test
trainPerc = 0.75;

% iterate over the filtered figure dirs
figDirs = dir(filteredAlignedImagesDir);
figDirs = figDirs(3:end);
nPersons = length(figDirs);

trueSamples = {};
falseSamples = {};
for iPerson = 1:nPersons
    fprintf('%d - %s\n', iPerson, figDirs(iPerson).name);
    
    % dir all images (not filtered)
    imagesDir = fullfile(alignedImagesDir, figDirs(iPerson).name);
    images = dir(fullfile(imagesDir, '*.jpg'));
    nImages = length(images);

    for iImage = 1:nImages
        imageName = images(iImage).name;
        imagePath = fullfile(imagesDir, imageName);
        
        % check for the same image in the filtered images
        filteredImagePath = fullfile(filteredAlignedImagesDir, figDirs(iPerson).name, imageName);
        if exist(filteredImagePath, 'file')
            % good image
            trueSamples{end+1} = imagePath;
        else
            % bad image
            % figure; imshow(imagePath);
            falseSamples{end+1} = imagePath;
        end
    end

end  

% write paths to files
nTrainTrue = round(trainPerc*length(trueSamples));
nTrainFalse = round(trainPerc*length(falseSamples));

fidTrain = fopen(outputFilePathTrain, 'w');
fidTest = fopen(outputFilePathTest, 'w');
for iImage = 1:length(trueSamples)
    if (iImage <= nTrainTrue)
        fprintf(fidTrain, '%s,1\n', trueSamples{iImage});
    else
        fprintf(fidTest, '%s,1\n', trueSamples{iImage});
    end
end
for iImage = 1:length(falseSamples)
    if (iImage <= nTrainFalse)
        fprintf(fidTrain, '%s,2\n', falseSamples{iImage});
    else
        fprintf(fidTest, '%s,2\n', falseSamples{iImage});
    end
end