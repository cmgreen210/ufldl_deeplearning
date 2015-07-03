function [cost,grad,features] = sparseAutoencoderLinearCost(theta, visibleSize, hiddenSize, ...
                                                            lambda, sparsityParam, beta, data)
% -------------------- YOUR CODE HERE --------------------
% Instructions:
%   Copy sparseAutoencoderCost in sparseAutoencoderCost.m from your
%   earlier exercise onto this file, renaming the function to
%   sparseAutoencoderLinearCost, and changing the autoencoder to use a
%   linear decoder.
% -------------------- YOUR CODE HERE --------------------                                    

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

m = size(data, 2);
m_inv = 1.0 / m;

z_2 = bsxfun(@plus, W1 * data, b1);
a_2 = sigmoid(z_2);

z_3 = bsxfun(@plus, W2 * a_2, b2);
a_3 = z_3; %LINEAR DECODER

rho_hat = m_inv .* sum(a_2, 2);
rho = repmat(sparsityParam, hiddenSize, 1);
kl = sum(rho .* log(rho ./ rho_hat) + (1-rho) .* log((1-rho)./(1-rho_hat)));

data_diff = data - a_3;

reg = (lambda / 2) * (sum(W1(:).^2) + sum(W2(:).^2));

cost = (m_inv/2) * sum((data_diff .* data_diff)(:)) + reg + beta * kl;


delta_3 = -1 * data_diff;
W2grad = m_inv * delta_3 * a_2' + lambda .* W2;
b2grad = m_inv * sum(delta_3, 2);

delta_2 = bsxfun(@plus, W2' * delta_3, beta .* (-1* rho./rho_hat + (1-rho)./(1-rho_hat))) .* a_2 .* (1 - a_2);
W1grad = m_inv * delta_2 * data' + lambda .* W1;
b1grad = m_inv * sum(delta_2, 2);

%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end
