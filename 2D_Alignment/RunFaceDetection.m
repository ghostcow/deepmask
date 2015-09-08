clearvars -except mainDir allImagesDir detectionsFileName;
close all;
warning('off', 'MATLAB:mir_warning_maybe_uninitialized_temporary');

currDir = fileparts(mfilename('fullpath'));
run(fullfile('detectors','voc-dpm','init.m'));

%%
mainDir = '/media/data/datasets/FaCeleb';
detectionsFileName = fullfile(mainDir, 'DPM_face_detections')
alignedImagesDir = fullfile(mainDir, 'aligned_scratch');
noFaceDir = fullfile(mainDir, 'aligned_scratch_no_face');
moreThanOneFaceDir = fullfile(mainDir, 'aligned_scratch_more_than_one_face');
shouldMove = true;

if ~exist(noFaceDir)
    mkdir(noFaceDir)
end
if ~exist(moreThanOneFaceDir)
    mkdir(moreThanOneFaceDir)
end

%% 
figDirs = dir(alignedImagesDir);
figDirs = figDirs([figDirs.isdir]); % clear all non dir files
figDirs(strncmp({figDirs.name}, '.', 1)) = []; % clear . and .. from dir
nPersons = length(figDirs);

temp = 1;
while true
    if ~exist([detectionsFileName '_' num2str(temp) '.txt'], 'file')
        detectionsFileName = [detectionsFileName '_' num2str(temp) '.txt'];
        break;
    end
    temp = temp + 1;
end

%%
fid = fopen(detectionsFileName, 'w');
for iFigure = 1:nPersons
    fprintf('%d - %s\n', iFigure, figDirs(iFigure).name);
    currDir = fullfile(alignedImagesDir, figDirs(iFigure).name);
    
    images = dir(fullfile(currDir, '*.jpg'));
    nImages = length(images);
    parfor iImage = 1:nImages  
        % original image
        imPath = fullfile(currDir, images(iImage).name);    
        im = imread(imPath);        
        try
            [ds, bs] = process_face(im, face_model.model,  ...
                                    detection_threshold, nms_threshold);
            if size(ds,1) == 1
                fprintf(fid, '%s\t%d\t%d\t%d\t%d\n', imPath, ds(1,1), ds(1,2), ds(1,3), ds(1,4));
            elseif size(ds,1) > 1
                if shouldMove
                    movefile(imPath, ...
                             fullfile(moreThanOneFaceDir, ...
                             [figDirs(iFigure).name '.' images(iImage).name]));
                end
            else
                if shouldMove
                    movefile(imPath, ...
                             fullfile(noFaceDir, ...
                             [figDirs(iFigure).name '.' images(iImage).name]));
                end
            end
        catch me
            fprintf('%s - Error - %s\n', images(iImage).name, me.message);
            continue;
        end   
    end
end
fclose(fid);