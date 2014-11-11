function [model] = get_EMLike_model(Xtrain,train_labs,params)
if ~exist('params','var') || isempty(params)
  params=[];
end
params=setParamsDefaults_impl(params,{...
    {'min_ids_img_num',2}...
    {'max_itter',300}...
    {'stop_th',1E-1}...
    {'initial_guess',struct('Se',[],'Sm',[])}...
    {'transfer_model',struct('Se',[],'Sm',[],'w',[])}...
    });

[d,N] = size(Xtrain);
if(isempty(params.initial_guess) || isempty(params.initial_guess.Se) || isempty(params.initial_guess.Sm))
  Sm = randn(d,10000);
  Sm = cov(Sm');
  Se = randn(d,10000);
  Se = cov(Se');
else
  Se = params.initial_guess.Se;
  Sm = params.initial_guess.Sm;
end

uids = unique(train_labs);
hist_ids = histc(train_labs,uids);
valid_ids = find(hist_ids >= params.min_ids_img_num);
iterI = 1;
stop_flag = false;
tr_pairsX = [];
ylike = [];
while (~stop_flag && iterI < params.max_itter)
  ti_w = tic;
  mu = {};
  ep = {};   
  F = inv(Se);
  ti = tic;
  for i=1:length(valid_ids)
    %E step:
    cur_imgs_inds = find(uids(valid_ids(i))==train_labs);
    cur_x = Xtrain(:,cur_imgs_inds);
    m = length(cur_imgs_inds);
    G = -(inv(m*Sm + Se))*Sm*F;
    cur_mu = sum(Sm*(F+m*G)*cur_x ,2);
    Seup = sum(Se*G*cur_x,2);
    cur_ep = bsxfun(@plus,cur_x,Seup);
    mu{i} = cur_mu;
    ep{i} = cur_ep;
    ylike(iterI,i) = 0;
    if(mod(i,50)==0)
      to=toc(ti);fprintf('i : %d/%d m : %d, %f(sec)\n',i,length(valid_ids),m,to);
      ti = tic;
    end
  end
  Sm_prev = Sm;
  Se_prev = Se;
  %M step:
  mu = cat(2,mu{:});
  ep = cat(2,ep{:});
  Sm = cov(mu');
  Se = cov(ep');
  if ~(isempty(params.transfer_model) || isempty(params.transfer_model.Se) || isempty(params.transfer_model.Sm))
    Sm = params.transfer_model.w*params.transfer_model.Sm + (1-params.transfer_model.w)*Sm;
    Se = params.transfer_model.w*params.transfer_model.Se + (1-params.transfer_model.w)*Se;
  end
  normSm(iterI) = norm(Sm-Sm_prev);
  normSe(iterI) = norm(Se-Se_prev);
  if(iterI>1)
    logLike_norm(iterI) = norm(diff(log(ylike(iterI-1:iterI,:))));
  else
    logLike_norm(iterI) = inf;
  end
  if(normSm(iterI) < params.stop_th && normSe(iterI) < params.stop_th)
    stop_flag = true;
  end
  to=toc(ti_w);fprintf('itter %d (%f sec) , normSm : %f, normSe : %f likelihood : %f normLike %f\n',iterI,to,normSm(iterI),normSe(iterI),mean(log(ylike(iterI,:))),logLike_norm(iterI));
  iterI = iterI + 1;
  if(isfield(params.initial_guess,'store_path'))
    save(params.initial_guess.store_path,'Se','Sm','-v7.3');
  end
end
model.Se = Se;
model.Sm = Sm;
model.mu = mu;
model.F = inv(Se);
model.G = -(inv(2*Sm+Se))*Sm*model.F;
model.A = inv(Sm+Se) - (model.F+model.G);

function [cur_mu,cur_ep,ylike] = EMloop(i,uids,valid_ids,train_labs,Xtrain,F,Se,Sm)
%E step:
cur_imgs_inds = find(uids(valid_ids(i))==train_labs);
cur_x = Xtrain(:,cur_imgs_inds);
m = length(cur_imgs_inds);
G = -(inv(m*Sm + Se))*Sm*F;
cur_mu = sum(Sm*(F+m*G)*cur_x ,2);
cur_ep = [];
for j=1:m
  cur_ep(:,j) = cur_x(:,j) + sum(Se*G*cur_x,2);
end
ylike = 0;
