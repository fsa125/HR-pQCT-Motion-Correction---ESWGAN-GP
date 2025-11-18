clc;
close all;
clear all;
% Load the image
image_path = uigetfile({'*.jpg;*.png;*.bmp;*.gif','All Image Files'; '*.*','All Files'}, 'Select an image file');
if image_path == 0
    disp('No image selected. Exiting...');
    return;
end

img = imread(image_path);

% Display the image
figure;
imshow(img);
title('Select regions to crop (double-click to finish cropping)');

cropped_images = cell(0, 1);  % Initialize cell array to store cropped images
crop_counter = 1;

% Loop to crop multiple regions
while true
    % Crop the image interactively
    h = imrect;
    wait(h);
    position = getPosition(h);
    
    % Check if user double-clicked outside the image to exit cropping
    if isempty(position)
        break;
    end
    
    % Crop the selected region
    cropped_img = imcrop(img, position);
    
    % Save cropped image
    [~, name, ext] = fileparts(image_path);
    cropped_image_name = [name '_cropped_' num2str(crop_counter) ext];
    imwrite(cropped_img, cropped_image_name);
    
    % Store cropped image in cell array
    cropped_images{end+1} = cropped_image_name;
    
    disp(['Cropped image ' num2str(crop_counter) ' saved as ' cropped_image_name]);
    
    % Increment crop counter
    crop_counter = crop_counter + 1;
end

% Display paths of cropped images
disp('Cropped images saved in the same folder:');
disp(cropped_images);