% clear variables;
close all; clc;
addpath('CEDD');

descDim = 144;
maxDist = 5;

mainDir = '/media/data/datasets/WLFDB'; 
alignedImagesDir = fullfile(mainDir, 'aligned_deepid');
dupsFileName = 'wlfdb_dups';
temp = 1;
while true
    if ~exist([dupsFileName '_' num2str(temp) '.txt'], 'file')
        dupsFileName = [dupsFileName '_' num2str(temp) '.txt'];
        break;
    end
    temp = temp + 1;
end

figDirs = dir(alignedImagesDir);
figDirs = figDirs([figDirs.isdir]); % clear all non dir files
figDirs(strncmp({figDirs.name}, '.', 1)) = []; % clear . and .. from dir
nPersons = length(figDirs);
fid = fopen(dupsFileName, 'w');
for iDir = 1:length(figDirs)
    fprintf('%d - %s\n', iDir, figDirs(iDir).name);
    figDir = fullfile(alignedImagesDir, figDirs(iDir).name);
%     figDir = fullfile(alignedImagesDir, '2552_Timothy_Olyphant');
    images = dir(fullfile(figDir, '*.jpg'));
    nImages = length(images);
    
    % compute descriptors
    descriptors = zeros(nImages, descDim);
    parfor iImage = 1:nImages
        % read image (handle some edge cases)
        imgPath = fullfile(figDir, images(iImage).name);
        [im, map] = imread(imgPath);
        X = size(im);
        if (length(X) > 3)
            % multi-frame image
            im = im(:,:,:,1);
        end
        if ~isempty(map)
            im = ind2rgb(im, map);
        end
        if (size(im, 3) == 1)
            % grayscale
            im = cat(3, im, im, im);
        end
        %
        descriptors(iImage, :) = CEDD(im)';
    end
    
    % compute all distances
    distMatrix = zeros(nImages, nImages);
    for iImage = 1:(nImages-1)
        for jImage = (iImage+1):nImages
            distMatrix(iImage, jImage) = Tanimoto(descriptors(iImage, :), descriptors(jImage, :));
            distMatrix(jImage, iImage) = distMatrix(iImage, jImage);
        end
    end
    
    % find dups
    dups = struct('baseIndex', {}, 'dupsIndices', {}, 'dists', {});
    isProcessed = false(1, nImages);
    for iImage = 1:(nImages-1)   
        distI = distMatrix(iImage, :);
        distI(iImage) = inf;
        distI(isProcessed) = inf;

        dupIndices = find(distI < maxDist);
        nDups = length(dupIndices);
        if (nDups > 0)
            % dups(end+1) = struct('baseIndex', iImage, 'dupsIndices', dupIndices, 'dists', distI(dupIndices));
            fprintf(fid, '%s,%d\n', fullfile(figDirs(iDir).name, images(iImage).name), nDups);
            for jImage = 1:length(dupIndices)
                fprintf(fid, '%s\n', fullfile(figDirs(iDir).name, images(dupIndices(jImage)).name));
            end
        end

        isProcessed(iImage) = true;
        isProcessed(dupIndices) = true;
    end
    
    % write into log the dups found
    if false
        % used for debug only
        for iDup = 1:length(dups)
            nDups = length(dups(iDup).dupsIndices);

            figure(iDup);
            figDim = ceil(sqrt(nDups+1));
            subplot(figDim, figDim, 1); imshow(fullfile(figDir, images(dups(iDup).baseIndex).name)); title('0');
            for iDupImage = 1:nDups
                subplot(figDim, figDim, 1+iDupImage); 
                imshow(fullfile(figDir, images(dups(iDup).dupsIndices(iDupImage)).name));
                title(num2str(dups(iDup).dists(iDupImage)));
            end
        end
    end
end
fclose(fid);