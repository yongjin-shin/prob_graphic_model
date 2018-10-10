%% HW01 EM
clear
close all
rng 'default'
%%

K = 3;
my_data = generator();
my_mog(my_data, K);

%%
function [area] = my_mog(data, K)
    area = solver(data, K);
    scatter(data(:,1), data(:,2), [], area(:, 1)); colorbar;
    scatter(data(:,1), data(:,2), [], area(:, 2)); colorbar;
    scatter(data(:,1), data(:,2), [], area(:, 3)); colorbar;

end

%%
function [area] = solver(sqz_d, K)
    tot = length(sqz_d);
    idx = randsample(tot, K);
    
    mean = [sqz_d(idx(1), :); sqz_d(idx(2), :); sqz_d(idx(3), :)];
    pi = ones(K,1); pi=pi*1/K;
    resp = get_resp(K, pi, sqz_d, tot, mean);
    
    iter = 0;
    max_iter = 100000;
    while(iter<max_iter)
        old_resp = resp;
        
        % M-STEP
        [mean, sigma, pi] = calculate(K, resp, sqz_d);
        
        % E-STEP
        resp = get_resp(K, pi, sqz_d, tot, mean, sigma);
        
        if abs(old_resp - resp) < 10^(-10)
            area = resp
            mean
            sigma
            pi
            break;
        else
            iter = iter+1;
            continue;
        end
    end    
end

%%
function [mean, sigma, pi] = calculate(K, resp, sqz_img)
    dim = ndims(sqz_img);

    %pi
    pi_sum = sum(resp, 1);
    tot_pi = sum(pi_sum, 2);
    pi = pi_sum/tot_pi; pi=pi';
    
    %mean
    mean = zeros(K,dim);
    for k=1:K
        for i = 1:dim
            mean(k,i) = sum(sqz_img(:,i).*resp(:,k),1)/pi_sum(k);
        end
    end
    
    %sigma
    sigma = zeros(dim,dim,K);
    for k=1:K
        r = resp(:, k); r = repmat(r, 1, dim);
        dif = sqz_img - mean(k,:);
        r_dif = dif.*r;
        sigma(:,:,k) = (r_dif'*dif)/pi_sum(k);
    end
    
    return
end

%%
function resp = get_resp(K, pi, sqz_img, tot, mean, sigma)
    dim = ndims(sqz_img);
    
    if nargin < 6
        sigma = eye(dim, dim); sigma = sigma*3;
        sigma = repmat(sigma, 1, 1, K);
    end
    
    y = zeros(tot, K);
    
    for k=1:K
        y(:,k) = mvnpdf(sqz_img(:,:), mean(k,:), sigma(:,:,k));
    end
    
    sum = y*pi;
    pi_long = repmat(pi', tot, 1);
    y = pi_long.*y;
    resp = y./sum;
    return 
end

%%
function [data] = generator()
    mu1 = [2 3];
    sigma1 = [1 1.5; 1.5 3];
    R1 = mvnrnd(mu1,sigma1,100);

    mu2 = [-1 -1];
    sigma2 = [2 -1.5; -1.5 2];
    R2 = mvnrnd(mu2,sigma2,60);

    mu3 = [4 1];
    sigma3 = [1 -1.5; -1.5 3];
    R3 = mvnrnd(mu3,sigma3,80);


    figure
    plot(R1(:,1),R1(:,2),'+')
    hold on
    plot(R2(:,1),R2(:,2),'+')
    hold on
    plot(R3(:,1),R3(:,2),'+')
    
    close all

    data = vertcat(R1, vertcat(R2, R3));
    data = data(randperm(length(data)), :);
    return
end
