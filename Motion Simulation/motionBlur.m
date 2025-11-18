function blurredImage = motionBlur (originalImage,kernel_size)

% Define motion blur kernel
kernelSize = kernel_size; % Adjust the kernel size based on the desired blur amount
motionBlurKernel = zeros(kernelSize);
motionBlurKernel(ceil(kernelSize/2), :) = 1/kernelSize;

% Apply motion blur using imfilter
blurredImage = imfilter(originalImage, motionBlurKernel, 'conv', 'replicate');

end