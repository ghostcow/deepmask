clear all;
close all;

currDir = fileparts(mfilename('fullpath'));
run(fullfile(currDir, 'init.m'));

%% Define all paths here
mainDir = '/media/data/datasets/pubfig';
allImagesDir = fullfile(mainDir, 'images');
alignedImagesDir = fullfile(mainDir, 'aligned');
detectionsFileName = 'detections_pubfig.txt';

% whether to take only the central face or all faces
useCenterFace = false;
isOverWrite = true;

%%
figDirs = dir(allImagesDir);
figDirs = figDirs([figDirs.isdir]); % clear all non dir files
figDirs(strncmp({figDirs.name}, '.', 1)) = []; % clear . and .. from dir
nPersons = length(figDirs);
noFacesCounter = 0;

fid = fopen(detectionsFileName, 'w');
for iFigure = 1:nPersons
    fprintf('%d - %s\n', iFigure, figDirs(iFigure).name);
    currDir = fullfile(allImagesDir, figDirs(iFigure).name);
    
    images = dir(fullfile(currDir, '*.jpg'));
    nImages = length(images);
    
    outputDir = fullfile(alignedImagesDir, figDirs(iFigure).name);
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end
        
    for iImage = 1:nImages  
        % original image
        imPath = fullfile(currDir, images(iImage).name);
        alignedImagePath = fullfile(outputDir, ...
            [images(iImage).name(1:end-3) 'jpg']);   
        if (~isOverWrite && exist(alignedImagePath, 'file'))
            continue;
        end
        try
            [detection, landmarks, aligned_imgs] = align_face(opts, imPath);
        catch me
            fprintf('%s - Error - %s\n', imPath, me.message);
        end

        nFaces = length(aligned_imgs);
        if (nFaces == 0)
            noFacesCounter = noFacesCounter + 1;
            continue;
        end
        
        if (nFaces == 1)
            imwrite(im2double(aligned_imgs{1}), alignedImagePath);   
            log_detections(fid, imPath, alignedImagePath, detection(:, 1));
        else
            if useCenterFace
                % look for the central face
                imInfo = imfinfo(imPath);
                middlePoint = [imInfo.Width / 2; imInfo.Height / 2];
                dists = bsxfun(@minus, detection(1:2, :), middlePoint);
                dists = sqrt(sum(dists.^2));
                [~, iCorrectFace] = min(dists);
                
                imwrite(im2double(aligned_imgs{iCorrectFace}), alignedImagePath); 
                log_detections(fid, imPath, alignedImagePath, detection(:, iCorrectFace));
            else
                % save all faces
                for iFace = 1:nFaces
                    alignedImageName = [images(iImage).name(1:end-4) '.' num2str(iFace) '.jpg'];
                    alignedImagePath = fullfile(outputDir, alignedImageName);
                    imwrite(im2double(aligned_imgs{iFace}), alignedImagePath);
                    log_detections(fid, imPath, alignedImagePath, detection(:, iFace));                  
                end
            end
        end
    end
end
fclose(fid);
fprintf('#pictures with no faces found = %d\n', noFacesCounter);