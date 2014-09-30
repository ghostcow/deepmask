clear all;
close all;

warning('off','MATLAB:iofun:UnsupportedEncoding');
currDir = fileparts(mfilename('fullpath'));
run(fullfile(currDir, '..', '..', 'img_preproc', 'init.m'));

mainDir = 'D:\_Dev\Datasets\Face Recognition\CFW';

filteredFacesDir = fullfile(mainDir, 'filtered_faces');
allImagesDir = fullfile(mainDir, 'images');
alignedImagesDir = fullfile(mainDir, 'filtered_aligned');
% gtFaces = ReadBbFile(fullfile(mainDir, 'bbox.txt'));
% save('bbox.mat', 'gtFaces');
load('bbox.mat', 'gtFaces');
detectionsFileName = 'detections.txt';

figDirs = dir(filteredFacesDir);
figDirs = figDirs(3:end);
nPersons = length(figDirs);
imSize = [152,152];

fid = fopen(detectionsFileName, 'a+');
for iFigure = 1:nPersons
    fprintf('%d - %s\n', iFigure, figDirs(iFigure).name);
    currDir = fullfile(filteredFacesDir, figDirs(iFigure).name);
    origImagesDir = fullfile(allImagesDir, figDirs(iFigure).name);
    images = dir(fullfile(currDir, '*.png'));
    nImages = length(images);
    
    for iImage = 1:nImages
        
        % original image
        imPath = fullfile(origImagesDir, [images(iImage).name(1:end-3) 'jpg']);
%         [im, map] = imread(fullfile(currDir, images(iImage).name));
%         if (size(im, 3) == 1)
%             if ~isempty(map)
%                 im = ind2rgb(im, map);
%             else
%                 error('grayscale image');
%             end
%         end
        
        try
            [detection, landmarks, aligned_imgs] = align_face(opts, imPath);
        catch me
            fprintf('%s - Error - %s\n', imPath, me.message);
            continue;
        end
        outputDir = fullfile(alignedImagesDir, figDirs(iFigure).name);
        if ~exist(outputDir, 'dir')
            mkdir(outputDir);
        end
        
        relImageName = fullfile(figDirs(iFigure).name, [images(iImage).name(1:end-3) 'jpg']);
        nFaces = length(aligned_imgs);
        iCorrectFace = 1;
        if (nFaces == 0)
            iCorrectFace = 0;
        else %if (nFaces > 1)
            % look for the correct face
            filteredFacePath = fullfile(filteredFacesDir, ...
                figDirs(iFigure).name, images(iImage).name);
            faceInfo = imfinfo(filteredFacePath);
            face = struct('height', faceInfo.Height,...
                'width', faceInfo.Width);
            if gtFaces.isKey(relImageName)

                faces = gtFaces(relImageName);
                iCorrespondFace = 1;
                for iFace = 2:length(faces)
                   if ((faces(iFace).height == face.height) && ...
                       (faces(iFace).width == face.width))
                        iCorrespondFace = iFace;
                   end
                end

                % choose the appropriate opencv detection
                minDist = 50; % 15 is the biggest distance allowed
                iCorrectFace = 0;
                for iFace = 1:nFaces
                    dist = norm(detection(1:2, iFace) - ...
                        faces(iCorrespondFace).center');
                    if (dist < minDist)
                        minDist = dist;
                        iCorrectFace = iFace;
                    end
                end

    %             if (minDist == 50)
    %                 figure; imshow(aligned_imgs{1}); 
    %                 title(sprintf('%s - %f', images(iImage).name, dist));
    %                 disp(images(iImage).name);
    %             end
            end
        end
        if (iCorrectFace ~= 0)
            alignedImagePath = fullfile(outputDir, ...
                [images(iImage).name(1:end-3) 'jpg']);
            imwrite(aligned_imgs{iCorrectFace}, alignedImagePath);
            fprintf(fid, '%s %d %d %d %d\n', relImageName, ...
                detection(1, iCorrectFace), detection(2, iCorrectFace), ...
                detection(3, iCorrectFace), detection(4, iCorrectFace));
        else
            fprintf('%s - no faces found\n', imPath);
        end
    end
end
fclose(fid);