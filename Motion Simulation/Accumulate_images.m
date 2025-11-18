% Set paths
main_folder = 'samples/Prox rad test img simz/';       % e.g., 'C:\Users\Farhan\Data\Volumes'
output_folder = fullfile(main_folder, 'combined_output');

% Create output folder if it doesn't exist
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

% Initialize global image counter
global_idx = 1;

% Loop over 90 volume folders
for v = 1:20 %volume numbers will change
    volume_folder = fullfile(main_folder, sprintf('volume_%d', v));
    
    % Get all image files (assume .png; change if needed)
    image_files = dir(fullfile(volume_folder, '*.png'));
    image_files = natsortfiles({image_files.name});  % Requires natsortfiles (optional)

    % Loop over all 168 images in the volume
    for i = 1:length(image_files)
        img_path = fullfile(volume_folder, image_files{i});
        img = imread(img_path);
        
        % Define output filename
        output_filename = sprintf('slice_%05d.png', global_idx);  % zero-padded
        imwrite(img, fullfile(output_folder, output_filename));
        
        global_idx = global_idx + 1;
        disp(global_idx)
    end
end