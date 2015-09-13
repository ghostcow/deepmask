clc; 
clearvars -except mainDir srcDir dstDir FLIP;
close all;

if ~exist('mainDir', 'var')
    mainDir = '/home/adampolyak/datasets/CASIA';
end

if ~exist('srcDir', 'var')
    srcDir = fullfile(mainDir, 'aligned_deepid');
end

if ~exist('dstDir', 'var')
    dstDir = fullfile(mainDir, 'aligned_nn');
end

if ~exist('FLIP', 'var')
    FLIP = true;
end

if ~exist(dstDir, 'dir')
    mkdir(dstDir);
end


figDirs = dir(srcDir);
figDirs = figDirs([figDirs.isdir]); % clear all non dir files
figDirs(strncmp({figDirs.name}, '.', 1)) = []; % clear . and .. from dir
nPersons = length(figDirs);

parfor iFigure = 1:nPersons
    fprintf('%d - %s\n', iFigure, figDirs(iFigure).name);
    
    currDir = fullfile(srcDir, figDirs(iFigure).name);
    currDstDir = fullfile(dstDir, figDirs(iFigure).name);
    images = dir(fullfile(currDir, '*.jpg'));
    nImages = length(images);
    
    if ~exist(currDstDir, 'dir')
        mkdir(currDstDir);
    end
    
    % make parfor
    for iImage = 1:nImages
        srcImagePath = fullfile(currDir, images(iImage).name);
        dstImagePath = fullfile(currDstDir, images(iImage).name);
        dstFlipedImagePath = fullfile(currDstDir, strcat('flipped_',images(iImage).name));
                
        im = imread(srcImagePath);
        % make rgb
        temp = zeros(size(im,1),size(im,2),3);
        if size(im,3) ~= 3
            temp(:,:,1) = im;
            temp(:,:,2) = im;
            temp(:,:,3) = im;
            im = temp;
        end
        % crop image
        im = im(28:269, 20:221, :);
        % resize image
        im = imresize(im, 0.7);
        
        imwrite(im2double(im), dstImagePath);
        if FLIP
            imwrite(im2double(flipdim(im,2)), dstFlipedImagePath);
        end
    end
end
