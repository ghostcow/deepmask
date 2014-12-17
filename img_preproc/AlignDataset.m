clear all variables;
close all;
warning('off', 'MATLAB:mir_warning_maybe_uninitialized_temporary');
addpath('../../mexopencv');
currDir = fileparts(mfilename('fullpath'));
type = 'deepid';
run(fullfile(currDir, 'init.m'));

fprintf('time : %s\n', datestr(now,'HH:MM'));
%% Define all constants here
mainDir = '/media/data/datasets/CASIA';
detectionsFileName = 'detections_casia';

allImagesDir = fullfile(mainDir, 'images');
alignedImagesDir = fullfile(mainDir, ['aligned_' type]);

useCenterFace = true; % whether to take only the central face or all faces
useFaceDetection = true; % use false, when the images are already contained centered faces

%% 
if useFaceDetection
    temp = 1;
    while true
        if ~exist([detectionsFileName '.txt'], 'file')
            break;
        else
            detectionsFileName = sprintf('%s_%d.txt', detectionsFileName, temp);
            temp = temp + 1;
        end
    end
    fid = fopen(detectionsFileName, 'w');
end

%%
figDirs = dir(allImagesDir);
figDirs = figDirs([figDirs.isdir]); % clear all non dir files
figDirs(strncmp({figDirs.name}, '.', 1)) = []; % clear . and .. from dir
nPersons = length(figDirs);
noFacesCounter = 0;

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
            if useFaceDetection
                [allFigDetections{iImage}{1}, landmarks, allFigDetections{iImage}{2}] = ...
                    align_face(opts, fullfile(currDir, images(iImage).name));
            else
                % use image borders as detection
                % detection = [x_center y_center width/2 1]
                imInfo = imfinfo(fullfile(currDir, images(iImage).name));
                allFigDetections{iImage}{1} = [imInfo.Width/2; imInfo.Height/2; imInfo.Width/2; 1];
                [~, landmarks, allFigDetections{iImage}{2}] = ...
                    align_face(opts, ...
                    fullfile(currDir, images(iImage).name), ...
                    struct('detections', allFigDetections{iImage}{1}));        
            end
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
            if useFaceDetection
                log_detections(fid, fullfile(currDir, images(iImage).name), ...
                    alignedImagePath, detection(:, 1)); 
            end
        else
            if useCenterFace
                % look for the central face
                imInfo = imfinfo(fullfile(currDir, images(iImage).name));
                middlePoint = [imInfo.Width / 2; imInfo.Height / 2];
                dists = bsxfun(@minus, detection(1:2, :), middlePoint);
                dists = sqrt(sum(dists.^2));
                [~, iCorrectFace] = min(dists);
                
                imwrite(im2double(aligned_imgs{iCorrectFace}), alignedImagePath); 
                if useFaceDetection
                    log_detections(fid, fullfile(currDir, images(iImage).name), ...
                        alignedImagePath, detection(:, iCorrectFace));
                end
            else
                % save all faces
                for iFace = 1:nFaces
                    alignedImageName = [images(iImage).name(1:end-4) '.' num2str(iFace) '.jpg'];
                    alignedImagePath = fullfile(outputDir, alignedImageName);
                    imwrite(im2double(aligned_imgs{iFace}), alignedImagePath);
                    if useFaceDetection
                        log_detections(fid, fullfile(currDir, images(iImage).name), ...
                            alignedImagePath, detection(:, iFace));
                    end
                end
            end
        end
    end
end
if useFaceDetection
    fclose(fid);
end
fprintf('#pictures with no faces found = %d\n', noFacesCounter);
fprintf('time : %s\n', datestr(now,'HH:MM'));