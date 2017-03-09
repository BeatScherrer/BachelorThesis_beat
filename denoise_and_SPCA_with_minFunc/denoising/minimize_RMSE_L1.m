function [lambda_min, rmse_min, lambda, rmse] = minimize_RMSE_L1(img_noisy,img, D, lambda)
    
rmse = zeros(size(lambda));

for i=1:size(lambda,2)
    img_denoised = simple_denoise_L1(img_noisy, lambda(i), D);
    rmse(i) = sqrt(sum((img(:) - img_denoised(:)).^2));
end

[rmse_min, i] = min(rmse);
lambda_min = lambda(i);

end