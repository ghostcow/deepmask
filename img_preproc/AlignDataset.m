clear all;
close all;

currDir = fileparts(mfilename('fullpath'));
run(fullfile(currDir, 'init.m'));

%% Define all paths here
mainDir = '/media/data/datasets/morph';
allImagesDir = fullfile(mainDir, 'images');
alignedImagesDir = fullfile(mainDir, 'aligned');

%%
figDirs = dir(allImagesDir);
figDirs = figDirs(3:end);
nPersons = length(figDirs);

for iFigure = 1:nPersons
    fprintf('%d - %s\n', iFigure, figDirs(iFigure).name);
    currDir = fullfile(allImagesDir, figDirs(iFigure).name);
    
    images = dir(currDir); %dir(fullfile(currDir, '*.jpg'));
    images = images(3:end);
    nImages = length(images);
    
    outputDir = fullfile(alignedImagesDir, figDirs(iFigure).name);
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end
        
    for iImage = 1:nImages  
        % original image
        imPath = fullfile(currDir, images(iImage).name);
        
        imPathNew = [imPath '.jpg'];
        movefile(imPath, imPathNew);
        images(iImage).name = [images(iImage).name '.jpg'];
        imPath = imPathNew;
        
        try
            [detection, landmarks, aligned_imgs] = align_face(opts, imPath);
        catch me
            fprintf('%s - Error - %s\n', imPath, me.message);
            continue;
        end

        nFaces = length(aligned_imgs);
        if (nFaces == 0)
            continue;
        end
        
        % saving only first face
        alignedImagePath = fullfile(outputDir, ...
            [images(iImage).name(1:end-3) 'jpg']);
        imwrite(im2double(aligned_imgs{1}), alignedImagePath);       
        if (nFaces > 1)
            for iFace = 2:nFaces
                alignedImagePath = fullfile(outputDir, ...
                    [images(iImage).name(1:end-4) '.' num2str(iFace) '.jpg']);
                imwrite(im2double(aligned_imgs{iFace}), alignedImagePath);
            end
        end
    end
end