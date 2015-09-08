clearvars -except mainDir allImagesDir detectionsFileName;
close all;
warning('off', 'MATLAB:mir_warning_maybe_uninitialized_temporary');

currDir = fileparts(mfilename('fullpath'));
type = 'deepid';
run(fullfile(currDir, 'init.m'));

%% Define all paths here
if ~exist('mainDir', 'var')
    mainDir = '/media/data/datasets/CASIA';
end

if ~exist('allImagesDir', 'var')
    allImagesDir = fullfile(mainDir, 'images');    
end

if ~exist('detectionsFileName', 'var')
    detectionsFileName = 'detections.txt';
end

alignedImagesDir = fullfile(mainDir, ['aligned_' type]);

temp = 1;
while true
    if ~exist(detectionsFileName, 'file')
        break;
    else
        detectionsFileName = sprintf('detections_%d.txt', temp);
        temp = temp + 1;
    end
end

% whether to take only the central face or all faces
useCenterFace = false;

%%
figDirs = dir(allImagesDir);
figDirs = figDirs([figDirs.isdir]); % clear all non dir files
figDirs(strncmp({figDirs.name}, '.', 1)) = []; % clear . and .. from dir
nPersons = length(figDirs);
noFacesCounter = 0;

% iStart = input('Please start index :');
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
        
    allFigDetections = cell(1, nImages);
    parfor iImage = 1:nImages  
        allFigDetections{iImage} = cell(2);
        try
            [allFigDetections{iImage}{1}, landmarks, allFigDetections{iImage}{2}] = ...
                align_face(opts, fullfile(currDir, images(iImage).name));
        catch me
            fprintf('%s - %d - Error - %s\n', ...
                figDirs(iFigure).name, iImage, me.message);
        end        
    end
    for iImage = 1:nImages
        alignedImagePath = fullfile(outputDir, ...
            [images(iImage).name(1:end-3) 'jpg']);   
        detection = allFigDetections{iImage}{1};
        aligned_imgs = allFigDetections{iImage}{2};

        nFaces = length(aligned_imgs);
        if (nFaces == 0)
            noFacesCounter = noFacesCounter + 1;
        elseif (nFaces == 1)
            imwrite(im2double(aligned_imgs{1}), alignedImagePath);  
            if size(detection,1)==0
                log_detections(fid, fullfile(currDir, images(iImage).name), ...
                alignedImagePath); 
            else
                log_detections(fid, fullfile(currDir, images(iImage).name), ...
                alignedImagePath, detection(:, 1)); 
            end
            
        else
            if useCenterFace
                % look for the central face
                imInfo = imfinfo(imPath);
                middlePoint = [imInfo.Width / 2; imInfo.Height / 2];
                dists = bsxfun(@minus, detection(1:2, :), middlePoint);
                dists = sqrt(sum(dists.^2));
                [~, iCorrectFace] = min(dists);
                
                imwrite(im2double(aligned_imgs{iCorrectFace}), alignedImagePath); 
                log_detections(fid, fullfile(currDir, images(iImage).name), ...
                    alignedImagePath, detection(:, iCorrectFace));
            else
                % save all faces
                for iFace = 1:nFaces
                    alignedImageName = [images(iImage).name(1:end-4) '.' num2str(iFace) '.jpg'];
                    alignedImagePath = fullfile(outputDir, alignedImageName);
                    imwrite(im2double(aligned_imgs{iFace}), alignedImagePath);
                    log_detections(fid, fullfile(currDir, images(iImage).name), ...
                        alignedImagePath, detection(:, iFace));
                end
            end
        end
    end
end
fclose(fid);
fprintf('#pictures with no faces found = %d\n', noFacesCounter);