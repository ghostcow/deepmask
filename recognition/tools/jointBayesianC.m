function [rw] = jointBayesianC(X1, X2, model)
% Use joint-bayesian model to classify feature pairs.
% 
% INPUTS : 
%   X1,X2 - featues matrices to compare, each of dimension [d x N]
%           in case X1 is only one feature than it's compared against all
%           features in X2
%   model - the joint-bayesian model (learned by calling jointBayesian)
% OUTPUTs : 
%   rw - classification scores for each pair (high value for match  & low
%       for mismatch) of the following form :
%       r(x1,x2) = log[P(x1,x2 | Hi) / P(x1,x2 | He)]
%       Hi is intra-personal variation hypothesis, while 
%       He is the extra-personal variation hypothesis

if (size(X1, 2) == 1)
    rw = X1'*model.A*X1 + diag(X2'*model.A*X2)' - 2*X1'*model.G*X2;
else
    rw = X1'*model.A*X1 + X2'*model.A*X2 - 2*X1'*model.G*X2;
    rw = diag(rw); %need only diag values of rw matrix.
end


