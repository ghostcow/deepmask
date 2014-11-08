clear all;
close all;

currDir = fileparts(mfilename('fullpath'));
run(fullfile(currDir, 'init.m'));

%% Define all paths here
mainDir = '/tmp/pubfig/images';
allImagesDir = mainDir;
alignedImagesDir = fullfile(mainDir, '../aligned_mexopencv');
detectionsFileName = '/tmp/pubfig/detections_pubfig_mexopencv.txt';

% whether to take only the central face or all faces
useCenterFace = false;
isOverWrite = true;

%%
figDirs = dir(allImagesDir);
figDirs = figDirs(4:end);
%nPersons = 3;
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
            [detection, landmarks, aligned_imgs, det2] = align_face(opts, imPath);
        catch me
            %fprintf('%s - Error - %s\n', imPath, me.message);
        end

        nFaces = length(aligned_imgs);
        if (nFaces == 0)
            noFacesCounter = noFacesCounter + 1;
            continue;
        end
        
        if (nFaces == 1)
            imwrite(im2double(aligned_imgs{1}), alignedImagePath);   
            fprintf(fid, '%s,%s\n', imPath, alignedImagePath);
            fprintf(fid, '%d %d %d %d\n', ...
                    detection(1, 1), detection(2, 1), ...
                    detection(3, 1), detection(4, 1));  
            fprintf(fid, '%d %d %d %d\n', ...
                    det2(1, 1), det2(2, 1), ...
                    det2(3, 1), det2(4, 1));  
        else
            if useCenterFace
                % look for the central face
                imInfo = imfinfo(imPath);
                middlePoint = [imInfo.Width / 2; imInfo.Height / 2];
                dists = bsxfun(@minus, detection(1:2, :), middlePoint);
                dists = sqrt(sum(dists.^2));
                [~, iCorrectFace] = min(dists);
                
                imwrite(im2double(aligned_imgs{iCorrectFace}), alignedImagePath);   
                fprintf(fid, '%s,%s,%d %d %d %d\n', imPath, alignedImagePath, ...
                        detection(1, iCorrectFace), detection(2, iCorrectFace), ...
                        detection(3, iCorrectFace), detection(4, iCorrectFace));  
            else
                % save all faces
                for iFace = 1:nFaces
                    alignedImagePath = fullfile(outputDir, ...
                        [images(iImage).name(1:end-4) '.' num2str(iFace) '.jpg']);
                    imwrite(im2double(aligned_imgs{iFace}), alignedImagePath);
                    fprintf(fid, '%s,%s\n', imPath, alignedImagePath);
                    fprintf(fid, '%d %d %d %d\n', ...
                            detection(1, iFace), detection(2, iFace), ...
                            detection(3, iFace), detection(4, iFace));  
                    fprintf(fid, '%d %d %d %d\n', ...
                            det2(1, iFace), det2(2, iFace), ...
                            det2(3, iFace), det2(4, iFace));                   
                end
            end
        end
    end
end
fclose(fid);
fprintf('#pictures with no faces found = %d\n', noFacesCounter);