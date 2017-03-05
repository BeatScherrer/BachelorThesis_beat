addpath(genpath('../libs'));

%%
img = im2double(imread('cameraman.tif'));
sz = size(img);
D = generate_first_difference_matrix_for_image(size(img));

Dx = D * img(:);
grad1 = Dx(1 : (sz(1)-1)*sz(2));
grad2 = Dx((sz(1)-1)*sz(2)+1:end);
grad1 = reshape(grad1, [sz(1)-1, sz(2)]);
grad2 = reshape(grad2, [sz(1), sz(2)-1]);

subplot(131);
imagesc(img);
subplot(132);
imagesc(grad1);
subplot(133);
imagesc(grad2);

%% Denoise with L_2 norm
img_noisy = img + randn(size(img)) * 0.2;
lambda = 1.1;

[lambda_min_L2, rmse_min_L2, lambda_L2, rmse_L2] = minimize_RMSE_L2(img_noisy,img,D);
img_denoised_L2 = simple_denoise_L2(img_noisy, lambda_min_L2, D);

subplot(131);
imagesc(img); caxis([0, 1]);
subplot(132);
imagesc(img_noisy); caxis([0, 1]);
subplot(133);
imagesc(img_denoised_L2); caxis([0, 1]);

rmse0 = sqrt(sum( (img(:) - img_noisy(:)).^2 ));
rmse1 = sqrt(sum( (img(:) - img_denoised_L2(:)).^2 ));
%fprintf('RMSE w/o denoising %.5f\nRMSE with L2 denoising %.5f\n', rmse0, rmse1);

%% Denoise with L_1 norm
img_noisy = img + randn(size(img)) * 0.2;
%lambda = 1.6;
[lambda_min_L1, rmse_min_L1, lambda_L1, rmse_L1] = minimize_RMSE_L1(img_noisy,img,D);
img_denoised_L1 = simple_denoise_L1(img_noisy, lambda_min_L1, D);
subplot(131);
imagesc(img); caxis([0, 1]);
subplot(132);
imagesc(img_noisy); caxis([0, 1]);
subplot(133);
imagesc(img_denoised_L1); caxis([0, 1]);

rmse0 = sqrt(sum( (img(:) - img_noisy(:)).^2 ));
rmse1 = sqrt(sum( (img(:) - img_denoised_L1(:)).^2 ));
%fprintf('RMSE w/o denoising %.5f\nRMSE with L1 denoising %.5f\n', rmse0, rmse1);
%% plot RMSE vs Lambda
subplot(121)
plot(lambda_L1, rmse_L1)
title('L_1 minimization')
xlabel('lambda')
ylabel('RMSE')
subplot(122)
plot(lambda_L2, rmse_L2)
title('L_2 minimization')
xlabel('lambda')
ylabel('RMSE')

fprintf('RMSE with L_1 = %f and L_2 = %f\n', rmse_min_L1, rmse_min_L2)
fprintf('lambda_L1 = %f and lambda_L2 = %f,\n',lambda_min_L1, lambda_min_L2)
%%
% TODO: vary lambda and watch RMSE values. Determine the best lambda value
% for given noise std.

% TODO2: implement with Total Variation regularization.
% The penalty term will be ||D*x||_1 instead of ||D*x||_2^2