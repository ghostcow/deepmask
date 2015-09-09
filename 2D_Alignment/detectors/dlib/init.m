clear opts;
detecorMainDir = fileparts(mfilename('fullpath'));
addpath(genpath(detecorMainDir));

opts = struct;
opts.predictorPath = fullfile(detecorMainDir,'shape_predictor_68_face_landmarks.dat');