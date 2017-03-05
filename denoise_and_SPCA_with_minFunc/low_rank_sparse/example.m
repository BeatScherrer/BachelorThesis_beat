addpath(genpath('..'));
% create random matrix of rank k
k = 5;
N = 100;
M = 200;
A = randn(N, M);
[U, S, V] = svd(A, 'econ');
S = diag(S);
S(k:end) = 0;
L = U * diag(S) * V';

% create random sparse matrix with p% elements == 0
p = 0.9;
S = randn(N, M);
Si = rand(N, M);
Si(Si < p) = 0;
S = S .* Si;

%
A = L + S;

%%
lambda = 0.05; % this is about right
% here you should modify the  low_rank_sparse_decomposition function and
% provide correct gradients
[L_est, S_est] = low_rank_sparse_decomposition(A, lambda);
frobS = norm(S - S_est, 'fro'); % Frobenius norm
frobL = norm(L - L_set, 'fro'); % both should be around 2 if implemented correctly
mean_error = (frobS + frobL) / 2;
%% TODO:
% evaluate decomposition accuracy wrt different values of k, p, and lambda
% produce three 2D error maps:
%            1) mean_error(k, lambda)
%            2) mean_error(p, lambda)
%            3) mean_error(k, p)
% choose grid size and spacing adequately for this experiments

