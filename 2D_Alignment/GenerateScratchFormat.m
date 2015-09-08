clc; 
clearvars -except mainDir srcDir dstDir FLIP;
close all;

if ~exist('mainDir', 'var')
    mainDir = '/media/data/datasets/CASIA';
end

if ~exist('srcDir', 'var')
    srcDir = fullfile(mainDir, 'aligned_deepid');
end

if ~exist('dstDir', 'var')
    dstDir = fullfile(mainDir, 'aligned_scratch');
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

for iFigure = 1:nPersons
    fprintf('%d - %s\n', iFigure, figDirs(iFigure).name);
    
    currDir = fullfile(srcDir, figDirs(iFigure).name);
    currDstDir = fullfile(dstDir, figDirs(iFigure).name);
    images = dir(fullfile(currDir, '*.jpg'));
    nImages = length(images);
    
    if ~exist(currDstDir, 'dir')
        mkdir(currDstDir);
    end
    
    % make parfor
    parfor iImage = 1:nImages
        srcImagePath = fullfile(currDir, images(iImage).name);
        dstImagePath = fullfile(currDstDir, images(iImage).name);
        dstFlipedImagePath = fullfile(currDstDir, strcat('flipped_',images(iImage).name));
                
        % crop the face, make gray scale, and resize
        im = imread(srcImagePath);
        if size(im,3) ~= 1
		grayIm = rgb2gray(im);
	else
		grayIm = im;
	end
        grayIm = grayIm(48:249, 20:221, :);
        grayIm = imresize(grayIm, 100/size(grayIm,2));
        
        imwrite(im2double(grayIm), dstImagePath);
        if FLIP
            imwrite(im2double(fliplr(grayIm)), dstFlipedImagePath);
        end
    end
end
