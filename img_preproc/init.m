clear opts;

cwd=cd;
cwd(cwd=='\')='/';
addpath([cwd '/utils']);
addpath([cwd '/facedet']);
addpath([cwd '/facefeats']);
addpath([cwd '/align']);

opts=load('facefeats/model.mat','model');

opts.alignparams.basePts=[25.0347   34.1802   44.1943   53.4623   34.1208   39.3564   44.9156   31.1454   47.8747 ;
                          34.1580   34.1659   34.0936   33.8063   45.4179   47.0043   45.3628   53.0275   52.7999];
opts.alignparams.scale=2;


