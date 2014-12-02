addpath('Model_3D/')
addpath('Model_2D/')

%% get landmarks for entire alignment process
threedee_landmarks = get_3D_model();
lfw_2D_landmarks = get_2D_landmarks();
cov_mat = cov(lfw_2D_landmarks);

%% for each image calculate camera and perform alignment
twodee_landmarks = [lfw_2D_landmarks(1, 1:2:end); ...
                    lfw_2D_landmarks(1, 2:2:end)];

P = get_camera(twodee_landmarks,threedee_landmarks, cov_mat);
[r, d] = get_residual(twodee_landmarks, threedee_landmarks, P);


%new_im = Part5_piecewise_affine(twodee_landmarks*2', ...
%                                threedee_landmarks(1:3,:)', ...
%                                im);
tform = fitgeotrans(twodee_landmarks'*2,r(1:2,:)','pwl');
im = imread('/Users/adamp/Research/test/lfw/aligned_deepid/John_Manley/John_Manley_0003.jpg');
new_im = imwarp(im,tform);

imshow(new_im);
