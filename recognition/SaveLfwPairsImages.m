outputFilePath = '../data_files/LFW/pairs.mat';
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
[namesLeft, namesRight, imgNumLeft, imgNumRight] = ParsePairsFile();
nPairs = length(namesLeft);
data = single(zeros(2*nPairs, 3, imSize(1), imSize(2)));
labels = uint16(zeros(2*nPairs, 1));
for iPair = 1:nPairs
%     figNameLeft = strrep(namesLeft{iPair}, ' ', '_');
    figNameRight = strrep(namesLeft{iPair}, ' ', '_');
    
    imPaths = {...
        fullfile(lfwAlignedImagesDir, namesLeft{iPair}, sprintf('%s_%04d.jpg', namesLeft{iPair}, imgNumLeft(iPair))), ...
        fullfile(lfwAlignedImagesDir, namesRight{iPair}, sprintf('%s_%04d.jpg', namesRight{iPair}, imgNumRight(iPair)))};
    imLabels = [mapNameToNum(figNameLeft), mapNameToNum(figNameRight)];
    for iImage = 1:2
        if ~exist(imPaths{iImage}, 'file')
            im = zeros(imSize(1), imSize(2), 3);
            % mark invalid images with label=0
            imLabels(iImage) = 0;
            fprintf('invalid image - %d (%s)\n', 2*(iPair-1)+iImage, imPaths{iImage});
        else
            [im, map] = imread(imPaths{iImage});
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

        data(2*(iPair-1)+iImage,:,:,:) = im;
        labels(2*(iPair-1)+iImage) = imLabels(iImage);
    end
end
save(outputFilePath, 'data', 'labels', '-v7.3');