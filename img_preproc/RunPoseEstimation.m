clc; clear all;
addpath('face-release1.0-basic');
run('face-release1.0-basic/init.m');

detectionsFileName = 'pose_lfw';
landmarksFileName = 'landmarks_lfw';
mainDir = '/media/data/datasets/LFW'; %'/media/data/datasets/WLFDB';
alignedImagesDir = fullfile(mainDir, 'lfw_aligned_deepid');
noFacesDir = fullfile(mainDir, 'no_face_pose');
nonfrontalDir = fullfile(mainDir, 'non_frontal_pose');
noncentralDir = fullfile(mainDir, 'non_central_face');
shouldMove = false;
%% 
figDirs = dir(alignedImagesDir);
figDirs = figDirs([figDirs.isdir]); % clear all non dir files
figDirs(strncmp({figDirs.name}, '.', 1)) = []; % clear . and .. from dir
nPersons = length(figDirs);

nNoFace = 0;
nNonFrontal = 0;
scaleFactor = 2;
faceCenterMaxShift = 30;

temp = 1;
while true
    if ~exist([landmarksFileName '_' num2str(temp) '.mat'], 'file')
        landmarksFileName = [landmarksFileName '_' num2str(temp) '.mat'];
        break;
    end
    temp = temp + 1;
end
temp = 1;
while true
    if ~exist([detectionsFileName '_' num2str(temp) '.txt'], 'file')
        detectionsFileName = [detectionsFileName '_' num2str(temp) '.txt'];
        break;
    end
    temp = temp + 1;
end

nLandmarksTot = 0;
landmarksSpace = 10;
landmarks = struct('path', cell(1, 0), 'bs', cell(1, 0));
save(landmarksFileName, 'landmarks');
landmarks = struct('path', cell(1, landmarksSpace), 'bs', cell(1, landmarksSpace));

fid = fopen(detectionsFileName, 'w');
for iFigure = 1:nPersons
    fprintf('%d - %s\n', iFigure, figDirs(iFigure).name);
    currDir = fullfile(alignedImagesDir, figDirs(iFigure).name);
    
    images = dir(fullfile(currDir, '*.jpg'));
    nImages = length(images);
    bs = cell(1, nImages);
    dimensions = zeros(nImages, 2);
    parfor iImage = 1:nImages  
        % original image
        imPath = fullfile(currDir, images(iImage).name);    
        im = imread(imPath);
        im = imresize(im, 1/scaleFactor);
        [nRows, nCols, nChannels] = size(im);
        dimensions(iImage, :) = [nRows, nCols];
        if (nChannels == 1)
            im = cat(3, im, im, im);
        end
        
        try
            bs{iImage} = detect(im, model, model.thresh);
            bs{iImage} = clipboxes(im, bs{iImage});
            bs{iImage} = nms_face(bs{iImage}, 0.3);
        catch me
            fprintf('%s - Error - %s\n', ...
                images(iImage).name, me.message);
            bs{iImage} = [];
            continue;
        end   
    end
    for iImage = 1:nImages  
        imPath = fullfile(currDir, images(iImage).name);    
        if (isempty(bs{iImage}))
            fprintf('No Face - %s\n', images(iImage).name);
            nNoFace = nNoFace + 1;
            if shouldMove
                movefile(imPath, fullfile(noFacesDir, [figDirs(iFigure).name '.' images(iImage).name]));
            end
        else
            bs{iImage} = bs{iImage}(1);
            % face location should be in the center (we assume the input images are already aligned
            centerMass = [mean(bs{iImage}.xy(:, 1:2), 2), mean(bs{iImage}.xy(:, 3:4), 2)];
            centerMass = mean(centerMass);
            if (norm(centerMass - [dimensions(iImage, 2) dimensions(iImage, 1)]/2) > faceCenterMaxShift);
                fprintf('No Central Face - %s\n', images(iImage).name);
                if shouldMove
                    movefile(imPath, fullfile(noncentralDir, [figDirs(iFigure).name '.' images(iImage).name]));
                end
            else
                % check pose
                pose = posemap(bs{iImage}.c);
                if (pose ~= 0)
                    fprintf('pose=%d - %s\n', pose, images(iImage).name);
                    nNonFrontal = nNonFrontal + 1;
                    if shouldMove
                        movefile(imPath, fullfile(nonfrontalDir, [figDirs(iFigure).name '.' images(iImage).name]));
                    end
                end
                % write into detections file : imPath,pose
                fprintf(fid, '%s,%d\n', imPath, pose);
                
                nLandmarksTot = nLandmarksTot + 1;
                landmarks(nLandmarksTot) = struct('path', imPath, 'bs', bs{iImage});
                if (nLandmarksTot == length(landmarks))
                    % landmarks array is full
                    %S = load(landmarksFileName);
                    %landmarks = [S.landmarks, landmarks];
                    
                    % save landmarks every few iterations (frequency is defined by landmarksSpace)
                    save(landmarksFileName, 'landmarks');
                    landmarks = [landmarks struct('path', cell(1, landmarksSpace), 'bs', cell(1, landmarksSpace))];
                end
            end
        end
    end
end
save(landmarksFileName, 'landmarks');
fclose(fid);