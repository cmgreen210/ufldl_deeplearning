function convolvedFeatures = cnnConvolve(patchDim, numFeatures, images, W, b, ZCAWhite, meanPatch)
%cnnConvolve Returns the convolution of the features given by W and b with
%the given images
%
% Parameters:
%  patchDim - patch (feature) dimension
%  numFeatures - number of features
%  images - large images to convolve with, matrix in the form
%           images(r, c, channel, image number)
%  W, b - W, b for features from the sparse autoencoder
%  ZCAWhite, meanPatch - ZCAWhitening and meanPatch matrices used for
%                        preprocessing
%
% Returns:
%  convolvedFeatures - matrix of convolved features in the form
%                      convolvedFeatures(featureNum, imageNum, imageRow, imageCol)

numImages = size(images, 4);
imageDim = size(images, 1);
numChannels = 3
imageChannels = size(images, numChannels);

convolvedFeatures = zeros(numFeatures, numImages, imageDim - patchDim + 1, imageDim - patchDim + 1);

patchDimSqrd = patchDim * patchDim;

WProc = W * ZCAWhite;
bProc = b - WProc * meanPatch;

for imageNum = 1:numImages
  for featureNum = 1:numFeatures

    % convolution of image with feature matrix for each channel
    convolvedImage = zeros(imageDim - patchDim + 1, imageDim - patchDim + 1);
    for channel = 1:3

      % Obtain the feature (patchDim x patchDim) needed during the convolution
      % ---- YOUR CODE HERE ----
      feature = WProc(featureNum, (1 + (channel - 1) * patchDimSqrd) : (channel * patchDimSqrd));
      feature = reshape(feature, patchDim, patchDim);

      % Flip the feature matrix because of the definition of convolution, as explained later
      feature = flipud(fliplr(squeeze(feature)));
      
      % Obtain the image
      im = squeeze(images(:, :, channel, imageNum));

      % Convolve "feature" with "im", adding the result to convolvedImage
      % be sure to do a 'valid' convolution
      % ---- YOUR CODE HERE ----

      convolvedImage = convolvedImage + conv2(im, feature, shape='valid');

    end
    
    % Subtract the bias unit (correcting for the mean subtraction as well)
    % Then, apply the sigmoid function to get the hidden activation
    % ---- YOUR CODE HERE ----

    convolvedImage = sigmoid(convolvedImage + bProc(featureNum));
    % ------------------------
    
    % The convolved feature is the sum of the convolved values for all channels
    convolvedFeatures(featureNum, imageNum, :, :) = convolvedImage;
  end
end


end

function out = sigmoid(x)
  out = 1 ./ (1 + exp(-x));
end

