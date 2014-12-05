%close all; 
clear variables; 
clc;

addpath('Model_3D/')
addpath('Model_2D/')
lfwDir = '/Users/adamp/Research/test/lfw/aligned_deepid/';

%% get landmarks for entire alignment process
threedee_landmarks = get_3D_model();
[lfw_2D_landmarks, lfwPaths] = get_2D_landmarks();
cov_mat = cov(lfw_2D_landmarks);

%% for each image calculate camera and perform alignment
iImage=1;
[temp, imageName, ext] = fileparts(lfwPaths{iImage});
[~, imageRelDir, ~] = fileparts(temp);
imPath = fullfile(lfwDir, imageRelDir, [imageName ext]);

twodee_landmarks = [lfw_2D_landmarks(1, 1:2:end); ...
                    lfw_2D_landmarks(1, 2:2:end)];

P = get_camera(twodee_landmarks,threedee_landmarks, cov_mat);
rXYZ = get_residual(twodee_landmarks, threedee_landmarks, P);

im = imread(imPath);
im = im2double(im);
xlimits = [29 208];
ylimits = [43 267];

twodee_landmarks = twodee_landmarks';
rXY = normalize_3D_model(rXYZ,xlimits,ylimits)';

% add image edges to avoid discontinuties
image_edges = get_image_edges(im);
twodee_landmarks = [twodee_landmarks;
                    image_edges];
rXY = [rXY;
       image_edges];

new_im = piecewise_affine(twodee_landmarks, ...
                          rXY, ...
                          im, ...
                          'visualize',true);
% tform = cp2tform(twodee_landmarks, xy_3d, 'piecewise linear');
% new_im = imtransform(im,tform);

figure;imshow(new_im);
