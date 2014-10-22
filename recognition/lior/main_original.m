LEFT        =normy2fast(LEFT);
RIGHT       =normy2fast(RIGHT);
LEFT_test   =normy2fast(LEFT_test);
RIGHT_test  =normy2fast(RIGHT_test);


X    =((LEFT      - RIGHT       ).^2) ./ (LEFT      + RIGHT      + 0.0001);
XTest=((LEFT_test - RIGHT_test  ).^2) ./ (LEFT_test + RIGHT_test + 0.0001);

%liblinear using C = 0.05

y       = 2*SNS     -1;
ytest   = 2*SNS_test-1;

C=0.05;

Model = CLSliblinear(sparse(double((X(:,ii)))),y(ii),3,C);
clsw = [Model.W Model.b]';
sc=[Model.W Model.b] * [(XTest(:,jj)); ones(1,length(jj))];
sctrain=[Model.W Model.b] * [(X); ones(1,size(X,2))];
wasflipped = mean(sign(sctrain)==y')<.5; 
if wasflipped
  sc = -sc; sctrain = -sctrain;
end

corrects = sign(sc)==ytest(jj)';
score = mean(corrects);


function X = normy2fast(X)

n = (sum(abs(X).^3)).^(1/3);
n(n<0.01) = 1;
X = bsxfun(@times,X,1./n);