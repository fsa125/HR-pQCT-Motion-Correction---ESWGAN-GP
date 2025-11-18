function imageStack = readImagesInFolder_2(folderPath,output_folder)
    % Get a list of all files in the folder
    files = natsortfiles(dir(fullfile(folderPath, '*.PNG'))); % Change '*.jpg' to match your image file format
    
    % Initialize an empty cell array to store the images
    imageStack = cell(1, numel(files));
    Z = zeros(1800, 1024, 168);
    % Loop through each image file and read it
    for i = 1:numel(files)
        % Construct the full file path
        filePath = fullfile(folderPath, files(i).name);
        
        % Read the image using imread
        img = imread(filePath);
        
        

        Z(:, :, i) = sino_2D(img);
        
        % Store the image in the cell array
        %imageStack{i} = img;
    end

    UDA_Recon_Z(Z,output_folder);
end