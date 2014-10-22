function [Model] = CLSliblinear(Xtrain,Ytrain,sPARAMS);

Xtrain = Xtrain';

if ~exist('sPARAMS','var')
    sPARAMS = [];
end

if ~isfield(sPARAMS,'C')
  sPARAMS.C = 1;
end

if ~isfield(sPARAMS,'additionalstring')
  sPARAMS.additionalstring = '';
end

if ~isfield(sPARAMS,'regression')
  sPARAMS.regression = 0;
end

if ~isfield(sPARAMS,'autobalance')
  sPARAMS.autobalance = 0;
end
if ~isfield(sPARAMS,'type')
  sPARAMS.type = 1;
end

if sPARAMS.C>=0,
  cstring = ['-c ' num2str(sPARAMS.C) ' '];
else
  cstring = '';
end

% auto-balancing added by Zohar
if sPARAMS.autobalance && max(Ytrain)==1 % perform auto-balancing only for 2-class
    nPos = sum(Ytrain==1);
    nNeg = length(Ytrain) - nPos;
    cMinus = (nNeg+nPos) / (2*nNeg);
    cPlus = (nNeg+nPos) / (2*nPos);
    weightstring = sprintf('-w-1 %f -w1 %f ', cMinus , cPlus);
else
    weightstring = '';
end

basicstring = sprintf('-q -s %d ', sPARAMS.type);

paramstring = [basicstring cstring weightstring ...
               sPARAMS.additionalstring];
Model.svmmodel = train(Ytrain,sparse(Xtrain),paramstring);

Model.paramstring = paramstring;
first = find(Ytrain ~= 0);
Model.FirstLabel = Ytrain( first(1) );


if isfield(sPARAMS,'saveflag'),
  r = 10; 
  save(sPARAMS.saveflag,'r');
end
