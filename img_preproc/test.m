clear all variables;
close all;

run('init.m');

%% get target coordinates of the 9 landmarks
[basePts, x0, x1, y0, y1] = GetAlignedImageCoords(opts.alignparams);
landmarksTarget = bsxfun(@minus, basePts, [x0;y0]) + 1;

% write landmarksTarget in to a txt file
dlmwrite('landmarks_target.txt', landmarksTarget);

%% align test image
imPath = 'test.jpg';
[detection, landmarks, aligned_imgs] = align_face(opts, imPath);
imwrite(aligned_imgs{1}, 'test_aligned.jpg');
figure; imshow(aligned_imgs{1}); hold on;
for iLandmark = 1:size(landmarksTarget, 2)
    plot(landmarksTarget(1, iLandmark), landmarksTarget(2, iLandmark), 'r+');
end
hold off;