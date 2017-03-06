function [L, S] = low_rank_sparse_decomposition(A, lambda)
% minimize ||x - img_noisy||_2^2 + lambda * ||D*x||_2^2
    % configure optimization
    opts = [];
    opts.display = 'iter'; % or off
    
    x0 = A(:)/2;
    objf = @(x) eval_cost_and_gradient(x, A, lambda);
    xmin = minFunc(objf, x0, opts);
    L = reshape(xmin, size(A));
    S = A - L;
end

function [f, grad] = eval_cost_and_gradient(x, A, lambda)
    L = reshape(x, size(A));
    S = A - L;
    
    [sU, sS, sV] = svd(L, 'econ');
    nuclear_norm = sum(diag(sS));
    l1_norm = sum(abs(S(:)));
    f = nuclear_norm + lambda * l1_norm;
    
    grad_nuclear_norm = sU * sV';
    error('Insert correct gradient here');
    grad_l1_norm = ; % ????
    grad = grad_nuclear_norm + lambda * grad_l1_norm;
    grad = grad(:); % everything should be stretched into vector
end