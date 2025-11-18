% -----------------------------------------------------------------------
% This file is part of the ASTRA Toolbox
% 
% Copyright: 2010-2022, imec Vision Lab, University of Antwerp
%            2014-2022, CWI, Amsterdam
% License: Open Source under GPLv3
% Contact: astra@astra-toolbox.com
% Website: http://www.astra-toolbox.com/
% -----------------------------------------------------------------------

%%This is the code for introducing synthetic physics-based motion
%%

function recon_2d = UDA_Recon(P,output_filename)
%Start from here!

%P -----------------------> image
% in-plane rotation angles
values = [pi/100, pi/120, -pi/60, -pi/40, -pi/20, pi/60, pi/50, pi/10, -pi/80];

numRandomValues = 1;  % You can change this to the desired number

% Generate random indices to select values from the list
randomIndices = randi(numel(values), 1, numRandomValues);

% Extract random values from the list using the indices
randomValues = values(randomIndices);


P = double (P);
P = imresize(P,[1024,1024]); %Resize it to [1024 1024]

angles = linspace2(-pi/4, 7*pi/4, 1800); %sinogram angles/spacing, 1800 -- number of angles in parallel beam geometry

angles_cor1 = angles + randomValues; %sinogram angles shifted - for a corrupted sinogram

vol_geom = astra_create_vol_geom(1024, 1024); 
proj_geom = astra_create_proj_geom('parallel', 1.0, 1024, angles);

vol_geom3 = astra_create_vol_geom(1024,1024); %previosuly 512 512
proj_geom3 = astra_create_proj_geom('parallel', 1.0, 1024, angles_cor1); %Number of detectors increase with higher pixels
%%

%%
[sinogram_id, sinogram] = astra_create_sino_gpu(P, proj_geom, vol_geom); %True sinogram
%[sinogram_id_og, sinogram_og] = astra_create_sino_gpu(P, proj_geom2, vol_geom2);
[sinogram_id_og1, sinogram_og1] = astra_create_sino_gpu(P, proj_geom3, vol_geom3); %corrupted sinogram

%% Randomly select 200 lines 
A =  randi([1, 1600]);
%B = randi([1, 900],1,800);
B = A + 200; %200


%for i =200 1:length(A)
%sinogram (A(i), :) = (1/sqrt(3)* cos (pi/6)*cos (pi/6))*sinogram_og(B(i),:);
%end
%for i = 300:320
%sinogram (i, :) = sinogram_og (i,:);
%end

%% Remove that specific 200 lines of the true sinogram with corrupted sinogram
for i = A:B
sinogram (i, :) = sinogram_og1 (i,:);
end


%%

%%

%%
astra_mex_data2d('delete', sinogram_id);

% We now re-create the sinogram data object as we would do when loading
% an external sinogram
sinogram_id = astra_mex_data2d('create', '-sino', proj_geom, sinogram);

% Create a data object for the reconstruction
rec_id = astra_mex_data2d('create', '-vol', vol_geom);

% Set up the parameters for a reconstruction algorithm using the GPU
cfg = astra_struct('SIRT_CUDA');
cfg.ReconstructionDataId = rec_id;
cfg.ProjectionDataId = sinogram_id;

% Available algorithms:
% SIRT_CUDA, SART_CUDA, EM_CUDA, FBP_CUDA (see the FBP sample)


% Create the algorithm object from the configuration structure
alg_id = astra_mex_algorithm('create', cfg);

% Run 150 iterations of the algorithm
astra_mex_algorithm('iterate', alg_id, 500);

% Get the result
rec = astra_mex_data2d('get', rec_id);
%figure(8); imshow(rec, []);
%figure(9); imshow(abs(rec-P) , []);
rec = uint8(rec);
imwrite(rec, output_filename);
% Clean up. Note that GPU memory is tied up in the algorithm object,
% and main RAM in the data objects.
astra_mex_algorithm('delete', alg_id);
astra_mex_data2d('delete', rec_id);
astra_mex_data2d('delete', sinogram_id);
end