function [W, PCAModel] = wpca(X, PCAModel, outputDim, sparams)
% Learn and project wightening PCA.
% given PCAModel project only, if not given learn it first.
% INPUTS : 
%   X - data, dimensions = [N x d]
%   PCAModel - when empty then this function learn the PCA model 
%   outputDim - desired output dim
%   sparams - struct with parameters (do_PCA_only = true/false)
%       when empty, default params are used
% OUTPUTS :
%   W - projected data
%   PCAModel - the learned model

if(~exist('sparams','var'))
  sparams = [];
end
if isempty(PCAModel)
    sparams.do_PCA_only = true;
else
    sparams.do_PCA_only = false;
end

if(issparse(X))
  X = full(X);
end
if(~exist('PCAModel','var')||isempty(PCAModel))
  fprintf(1,'WPCA train...');
  if(1)
    ti=tic;
    avg = mean(X,1);
    mX =  bsxfun(@minus,X,avg);
    
    [COEFF,score,latent,tsquare] = princomp(mX,'econ');
    if(sparams.do_PCA_only)
      lamdas_factor = 1;
    else
      lamdas_factor = diag(1./(sqrt(latent)));
    end
    new_COEFF = (COEFF*lamdas_factor);
    coeff = new_COEFF(:,1:min(outputDim,size(new_COEFF,2)));
  end
  
  PCAModel.mean = avg;
  PCAModel.full_proj = coeff;
  to=toc(ti);
  fprintf(1,'PCA Learning time is : %f sec \n',to);                       
end %of train

mX = (bsxfun(@minus,X,PCAModel.mean));
W = (mX*PCAModel.full_proj);

