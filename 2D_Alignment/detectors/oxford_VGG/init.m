clear opts;
detecorMainDir = fileparts(mfilename('fullpath'));
addpath(genpath(detecorMainDir));

opts=load(fullfile(detecorMainDir, 'facefeats', 'model.mat'),'model');