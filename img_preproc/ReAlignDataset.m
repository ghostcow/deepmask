clear variables;
close all;

currDir = fileparts(mfilename('fullpath'));
type = 'deepid';
run(fullfile(currDir, 'init.m'));
addpath(fullfile(currDir, '..', 'data_set'));
saveLandmarks = true; % set true only in the first run !

%% Load all existing datasets info
type = 'deepface'; % load metadata for original deepface aligned images
LoadAllDatasetsPaths;
% 1 - CFW, 2 - CFW_small2, 3 - PubFig, 4 - SUFR, 5 - missing_cfw_pubfig (google), 6 - missing_sufr (google)
outputDirs = {'/media/data/datasets/CFW/aligned_deepid', ...
    '/media/data/datasets/CFW/aligned_deepid', ...
    '/media/data/datasets/pubfig/aligned_deepid', ...
    '/media/data/datasets/SUFR/aligned_deepid', ...
    '/media/data/datasets/missing_pubfig_cfw/aligned_deepid', ...
    '/media/data/datasets/missing_sufr/aligned_deepid', ...
    '/media/data/datasets/LFW/lfw_aligned_deepid'};

%% Define all paths here
for iDir = 7:length(mainDirs)
    mainDir = mainDirs{iDir};
    fprintf('%d : %s\n', iDir, mainDir);
    outputDir = outputDirs{iDir};
    
    if saveLandmarks
        landmarksFilePath = fullfile(fileparts(outputDir), 'landmarks.mat');
        if exist(landmarksFilePath, 'file')
            % landmarks file already exist for this db
            S = load(landmarksFilePath);
            landmarksTot = S.landmarksTot;
            iImageTot = length(landmarksTot) + 1;
        else
            landmarksTot = struct('imPath', cell(1, 80000), 'alignedPath', cell(1, 80000), 'landmarks', cell(1, 80000));
            iImageTot = 1;
        end
    end
    
    % start iterating
    figDirs = dir(mainDir);
    figDirs = figDirs(3:end);
    nPersons = length(figDirs);
    for iPerson = 1:nPersons
        fprintf('%d : %s\n', iPerson, figDirs(iPerson).name);
        imagesDir = fullfile(mainDir, figDirs(iPerson).name);
        images = dir(fullfile(imagesDir, '*.jpg'));
        nImages = length(images);
        outputDirPerson = fullfile(outputDir, figDirs(iPerson).name);
        if ~exist(outputDirPerson, 'dir')
            mkdir(outputDirPerson);
        end
        
        % looking for the original image
        % sort paths by face resolution
        for iImage = 1:nImages
            key = fullfile(figDirs(iPerson).name, images(iImage).name);
            if ~isKey(detections{iDir}, key)
                fprintf('WARNING! aligned images with no detection : %s\n', images(iImage).name);
            else
                currImageDetection = detections{iDir}(key);
                alignedImagePath = fullfile(outputDirPerson, images(iImage).name);
                
                % run faster face alignment by supplting the detection
                try
                    [~, landmarks, aligned_imgs] = align_face(opts, ...
                        currImageDetection.path, struct('detections', currImageDetection.detection));
                    imwrite(aligned_imgs{1}, alignedImagePath);
                    if saveLandmarks
                        landmarksTot(iImageTot) = struct('imPath', currImageDetection.path, ...
                            'alignedPath', key, 'landmarks', landmarks{1});
                        iImageTot = iImageTot + 1;
                    end
                catch me
                    fprintf('error in align_face : %s, %s\n', me.message, currImageDetection.path);
                end
            end
        end
    end
    
    if saveLandmarks
        landmarksTot(iImageTot:end) = [];
        save(landmarksFilePath, 'landmarksTot');
    end
end