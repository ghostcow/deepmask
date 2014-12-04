close all; 
clear variables; clc;
addpath('Model_3D/')
addpath('Model_2D/')
% lfwDir = '/Users/adamp/Research/test/lfw/aligned_deepid/';
lfwDir = '/media/data/datasets/LFW/lfw_aligned_deepid/';

figure; 
xyz_model = display_3d_model();	
height = 100; width = 100;

% take (x,z) coordinates
xy_model = xyz_model([1 3],:);
% normalize x,y coordinates from the 3D model
xy_min = min(xy_model, [], 2);
xy_max = max(xy_model, [], 2);
xy_model_normalized = bsxfun(@minus, xy_model, xy_min);
xy_model_normalized = bsxfun(@times, xy_model_normalized, 1 ./ (xy_max - xy_min));
xy_model_normalized = 1-xy_model_normalized; % vertical flip yo x,y coordinates

xy_model = bsxfun(@times, xy_model_normalized, [100 ; 100]);
figure; imshow(zeros(100, 100)); hold;
plot(xy_model(1,:), xy_model(2,:), 'r.');
labels=cellstr(num2str([1:size(xy_model, 2)]'));
text(xy_model(1,:), xy_model(2,:), labels, 'Color', 'w');
hold;

%% get landmarks for entire alignment process
threedee_landmarks = get_3D_model();
[lfw_2D_landmarks, lfwPaths] = get_2D_landmarks();
cov_mat = cov(lfw_2D_landmarks);

%% for each image calculate camera and perform alignment
iImage = 1;
[temp, imageName, ext] = fileparts(lfwPaths{iImage});
[~, imageRelDir, ~] = fileparts(temp);
imPath = fullfile(lfwDir, imageRelDir, [imageName ext]);

twodee_landmarks = [lfw_2D_landmarks(iImage, 1:2:end); ...
                    lfw_2D_landmarks(iImage, 2:2:end)];

P = get_camera(twodee_landmarks,threedee_landmarks, cov_mat);
[r, d] = get_residual(twodee_landmarks, threedee_landmarks, P);

%new_im = Part5_piecewise_affine(twodee_landmarks*2', ...
%                                threedee_landmarks(1:3,:)', ...
%                                im);

im = imread(imPath);
[height, width, ~] = size(im);
im = im2double(im);

% option 1 - match to 3D points on camera ray (closest to model points)
% take (x,z) coordinates
xy_3d = r([1 3],:); 
% normalize coordinates
xy_3d_min = min(xy_3d, [], 2);
xy_3d_max = max(xy_3d, [], 2);
xy_3d = bsxfun(@minus, xy_3d, xy_3d_min);
xy_3d = bsxfun(@times, xy_3d, 1 ./ (xy_3d_max - xy_3d_min));
xy_3d = 1 - xy_3d; % vertical flip of  x,y coordinates
% scale points to fixed range
xlimits = [29 208];
ylimits = [43 267];

xy_3d = bsxfun(@times, xy_3d, [xlimits(2) - xlimits(1) + 1 ; ylimits(2) - ylimits(1) + 1]);
xy_3d = bsxfun(@plus, xy_3d, [xlimits(1) ; ylimits(1)]);

twodee_landmarks = twodee_landmarks';
xy_3d = xy_3d';

% add image corners
twodee_landmarks = [twodee_landmarks;
    1 1; width 1; width height; 1 height];
xy_3d = [xy_3d;
    1 1; width 1; width height; 1 height];

% add some points on the image borders
horizonPoints = round(width/5):round(width/5):round(width*4/5);
verticalPoints = round(height/7):round(height/7):round(height*6/7);
twodee_landmarks = [twodee_landmarks;
    horizonPoints', ones(length(horizonPoints), 1); ...
    horizonPoints', height*ones(length(horizonPoints), 1); ...
    ones(length(verticalPoints), 1), verticalPoints'; ...
    width*ones(length(verticalPoints), 1), verticalPoints'];
xy_3d = [xy_3d; ...
    horizonPoints', ones(length(horizonPoints), 1); ...
    horizonPoints', height*ones(length(horizonPoints), 1); ...
    ones(length(verticalPoints), 1), verticalPoints'; ...
    width*ones(length(verticalPoints), 1), verticalPoints'];

figure; 
subplot(1, 2, 1);
imshow(im); hold;
plot(twodee_landmarks(1:length(labels), 1), twodee_landmarks(1:length(labels), 2), '.');
text(twodee_landmarks(1:length(labels), 1), twodee_landmarks(1:length(labels), 2), labels, 'Color', 'g');
subplot(1, 2, 2);
imshow(zeros(size(im)));
text(xy_3d(1:length(labels), 1), xy_3d(1:length(labels), 2), labels, 'Color', 'g');

% new_im1 = piecewise_affine(twodee_landmarks, xy_3d, im, 'visualize', false);
tform1 = cp2tform(twodee_landmarks, xy_3d, 'piecewise linear');
new_im1 = imtransform(im,tform1);
% figure; imshow(new_im1);

figure; 
imshow(im); hold;
plot(twodee_landmarks(:, 1), twodee_landmarks(:, 2), 'r.');
hold;
title('original with 2D landmarks');

figure;
subplot(1,2,1); 
imshow(zeros(height, width)); hold;
plot(xy_3d(:, 1), xy_3d(:, 2), 'b.');
hold;
title('3D landmars target');

subplot(1,2,2); 
imshow(new_im1, []); hold;
plot(xy_3d(:, 1), xy_3d(:, 2), 'b.');
hold;
title('aligned');