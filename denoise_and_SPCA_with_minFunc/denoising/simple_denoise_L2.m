function img_denoised = simple_denoise_L2(img_noisy, lambda, D)
% minimize ||x - img_noisy||_2^2 + lambda * ||D*x||_2^2
    % configure optimization
    opts = [];
    opts.display = 'iter'; % or off
    
    x0 = img_noisy(:);
    % we should create a function that receives vector x and returns value
    % with gradient. All data handling is encapsulated in lambda-expression
    % (has nothing to do with parameter "lambda" name)
    objf = @(x) eval_cost_and_gradient(x, img_noisy, lambda, D);
    xmin = minFunc(objf, x0, opts);
    img_denoised = reshape(xmin, size(img_noisy));
end

function [f, grad] = eval_cost_and_gradient(x, img_noisy, lambda, D)
    Dx = D * x;
    f = sum((x - img_noisy(:)).^2)/2 + lambda/2 * sum(Dx.^2); 
    grad = x - img_noisy(:) + lambda * (D' * Dx); 
end