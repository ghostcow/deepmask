function X = normy2fast(X, p)

if ~exist('p', 'var')
    p = 3;
end

n = (sum(abs(X).^p)).^(1/p);
n(n<0.01) = 1;
X = bsxfun(@times,X,1./n);