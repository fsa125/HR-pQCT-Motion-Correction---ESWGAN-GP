function imageStack = readImagesInFolder(folderPath,output_folder)
    % Get a list of all files in the folder
    files = natsortfiles(dir(fullfile(folderPath, '*.PNG'))); % Change '*.jpg' to match your image file format
    
    % Initialize an empty cell array to store the images
    imageStack = cell(1, numel(files));
    
    % Loop through each image file and read it
    for i = 1:numel(files)
        % Construct the full file path
        filePath = fullfile(folderPath, files(i).name);
        
        % Read the image using imread
        img = imread(filePath);
        
        filename = sprintf('%d.png', i);

        fullPath = fullfile(output_folder, filename);

        UDA_Recon (img, fullPath );
        
        % Store the image in the cell array
        %imageStack{i} = img;
    end
end