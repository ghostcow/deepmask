clc; clear variables;
outputFilePath = '../data_files/LFW/people.mat';
lfwAlignedImagesDir = '/media/data/datasets/LFW/lfw_aligned';
imSize = [152, 152];

% convert lfw names to numbers (directories indices)
figDirs = dir(lfwAlignedImagesDir);
figDirs = figDirs(3:end);
figDirs = {figDirs.name};
mapNameToNum = containers.Map;
for iDir = 1:length(figDirs)
    mapNameToNum(figDirs{iDir}) = iDir;
end

% loading all images
peopleMetadata = GetPeopleData();

maxNumImages = 30000;
data = single(zeros(maxNumImages, 3, imSize(1), imSize(2)));
labels = uint16(zeros(maxNumImages, 1));
splitId = uint16(zeros(maxNumImages, 1));
iImageGlobal = 1;
for iFold = 1:numel(peopleMetadata)
    currFold = peopleMetadata{iFold};
    for iPerson = 1:numel(currFold)
        personName = currFold(iPerson).name;
        nImages = length(currFold(iPerson).imageIndices);
        labels(iImageGlobal:(iImageGlobal + nImages - 1)) = mapNameToNum(personName);
        splitId(iImageGlobal:(iImageGlobal + nImages - 1)) = iFold;
        for iImage = 1:nImages
            imPath = fullfile(lfwAlignedImagesDir, personName, sprintf('%s_%04d.jpg', personName, iImage));
            if ~exist(imPath, 'file')
                im = zeros(imSize(1), imSize(2), 3);
                % mark invalid images with label=0
                labels(iImageGlobal + iImage - 1) = 0;
                fprintf('invalid image - %s\n', sprintf('%s_%04d.jpg', personName, iImage));
            else
                [im, map] = imread(imPath);
                if ~isempty(map)
                    im = ind2rgb(im, map);
                end
                if (size(im, 3) == 1)
                    im = cat(3, im, im, im);
                end
            end
            im = im2single(im);
            % convert shape from 152x152x3 3x152x152
            im = shiftdim(im, 2); 

            data(iImageGlobal + iImage - 1,:,:,:) = im;
        end
        iImageGlobal = iImageGlobal + nImages;
    end
end
data(iImageGlobal:end, :, :, :) = [];
labels(iImageGlobal:end) = [];
splitId(iImageGlobal:end) = [];
save(outputFilePath, 'data', 'labels', 'splitId', '-v7.3');