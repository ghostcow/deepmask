addpath('align');
run(fullfile('detectors','dlib','init.m'));
type = 'deepid'
%run(fullfile('detectors','oxford_VGG','init.m'));
%run(fullfile('detectors','ZhuRaman','init.m'));

% base locations for facial keypoints
opts.alignparams.basePts=[25.0347   34.1802   44.1943   53.4623   34.1208   39.3564   44.9156   31.1454   47.8747 ;
                          34.1580   34.1659   34.0936   33.8063   45.4179   47.0043   45.3628   53.0275   52.7999];

if (strcmp(type, 'deepface'))  
    % tight face, size 152 x 152
    opts.alignparams.scale = 3.5;
    opts.alignparams.horRatio = 1.12;
    opts.alignparams.topRatio = 0.78;
    opts.alignparams.bottomRatio = 0.5;
elseif (strcmp(type, 'deepid'))  
    % params for deepID - not too tight face cropping
    opts.alignparams.scale = 3.5; % to get 152x152 aligned image : im = aligned(78:229, 50:201, :)
    opts.alignparams.horRatio = 1.7;
    opts.alignparams.topRatio = 2;
    opts.alignparams.bottomRatio = 1.2;
end