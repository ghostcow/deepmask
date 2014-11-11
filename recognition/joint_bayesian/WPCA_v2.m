function [W,PCAModel] = WPCA_v2(X,PCAModel,OutdimNum,params)
%Learn and project wightening PCA.
%given PCAModel project only, if not given learn it first.
%X is (NXd)

if(~exist('params','var'))
  params = [];
end
if isempty(PCAModel)
    params.do_PCA_only = true;
else
    params.do_PCA_only = false;
end
% params=setParamsDefaults_impl(params,{...
%     {'do_PCA_only',false}...
%     });


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
    if(params.do_PCA_only)
      lamdas_factor = 1;
    else
      lamdas_factor = diag(1./(sqrt(latent)));
    end
    new_COEFF = (COEFF*lamdas_factor);
    coeff = new_COEFF(:,1:min(OutdimNum,size(new_COEFF,2)));
  end
  
  PCAModel.mean = avg;
  PCAModel.full_proj = coeff;
  to=toc(ti);fprintf(1,'PCA Learning time is : %f sec ',to);                       
end %of train

mX = (bsxfun(@minus,X,PCAModel.mean));
W = (mX*PCAModel.full_proj);

