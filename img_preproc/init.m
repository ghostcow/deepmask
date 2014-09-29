clear opts;
mainDir = fileparts(mfilename('fullpath'));
addpath(genpath(mainDir));

opts=load(fullfile(mainDir, 'facefeats', 'model.mat'),'model');

% base locations for facial keypoints
opts.alignparams.basePts=[25.0347   34.1802   44.1943   53.4623   34.1208   39.3564   44.9156   31.1454   47.8747 ;
                          34.1580   34.1659   34.0936   33.8063   45.4179   47.0043   45.3628   53.0275   52.7999];

% face only, size 152 x 152
opts.alignparams.scale = 3.5;
opts.alignparams.horRatio = 1.12;
opts.alignparams.topRatio = 0.78;
opts.alignparams.bottomRatio = 0.5;