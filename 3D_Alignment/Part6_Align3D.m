addpath('Part1_3D_Model/')
addpath('Part2_covariance/')

%% get landmarks for entire alignment process
threedee_landmarks = display_3d_model();
[twodee_landmarks, cov_mat] = get_cov_mat('landmarks_lfw.mat');

%% for each image calculate camera and perform alignment
P = Part3_get_camera(twodee_landmarks(:,1),threedee_landmarks, cov_mat);
[r, d] = Part4_get_residual(twodee_landmarks(:,1), threedee_landmarks, P);

im = imread('/Users/adamp/Research/test/lfw/aligned_deepid/John_Manley/John_Manley_0003.jpg');
twodee_landmarks = [twodee_landmarks(1:2:end, 1)'; ...
                    twodee_landmarks(2:2:end, 1)']';
%new_im = Part5_piecewise_affine(twodee_landmarks*2, ...
%                                threedee_landmarks(:,1:2), ...
%                                im);
try
tform = fitgeotrans(twodee_landmarks*2,threedee_landmarks(:,1:2),'pwl');
catch
end
new_im = imwarp(im,tform);
imshow(new_im);
