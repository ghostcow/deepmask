function [rw] = classify_bayes_joint_form(X1,X2,model)

rw = X1'*model.A*X1 + X2'*model.A*X2 - 2*X1'*model.G*X2;
rw = diag(rw);%need only diag values of rw matrix.
