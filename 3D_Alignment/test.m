function [ref_XY] = test()
addpath('/Users/adamp/Research/Matlab/3D_Alignment/face-release1.0-basic/');
% load and visualize model
% Pre-trained model with 146 parts. Works best for faces larger than 80*80
load face_p146_small.mat

% 5 levels for each octave
model.interval = 5;
% set up the threshold
model.thresh = min(-0.65, model.thresh);

% define the mapping from view-specific mixture id to viewpoint
if length(model.components)~=13 && length(model.components)~=18
    error('Can not recognize this model');
end

figure;
im = imread('/Users/adamp/Research/test/lfw/aligned_deepid/John_Manley/John_Manley_0003.jpg'); 
hold on; 

t = detect(im, model, -0.65); 
t = clipboxes(im, t);
t = nms_face(t,0.3);


x1 = t(1).xy(:,1);  
y1 = t(1).xy(:,2);  
x2 = t(1).xy(:,3);  
y2 = t(1).xy(:,4);
ref_XY = [(x1+x2)/2,(y1+y2)/2];

i=1;
j=68;
imshow(im); 
hold on; 
plot(ref_XY(i:j,1), ref_XY(i:j,2), 'r.');
hold on;
labels=cellstr(num2str([i:j]'));
text(ref_XY(i:j,1), ref_XY(i:j,2), labels);