% -----------------------------------------------------------------------
% This file is part of the ASTRA Toolbox
% 
% Copyright: 2010-2022, imec Vision Lab, University of Antwerp
%            2014-2022, CWI, Amsterdam
% License: Open Source under GPLv3
% Contact: astra@astra-toolbox.com
% Website: http://www.astra-toolbox.com/
% -----------------------------------------------------------------------

%%This is the code for introducing synthetic physics-based motion in the Z
%%direction
%%

function recon_2d = sino_2D(P)

%Start from here!
% List of 4 values
%ESWGAN-GP
%values = [pi/100, pi/120, -pi/60, -pi/40, -pi/20, pi/60, pi/50, pi/10, -pi/80]; 
%Transformer
values = [pi/100, pi/120, -pi/60, -pi/40, pi/60, pi/50, -pi/80];
%values = [0]; %for target
%values = [-pi/120];
% Number of random values to generate
numRandomValues = 1;  % You can change this to the desired number

% Generate random indices to select values from the list
randomIndices = randi(numel(values), 1, numRandomValues);

% Extract random values from the list using the indices
randomValues = values(randomIndices);
%vol_geom = astra_create_vol_geom(168, 256, 256);
%vol_geom_1 = astra_create_vol_geom(168, 256, 256);
%PSF = fspecial('motion',30,0);
%blurred = imfilter(Idouble,PSF,'conv','circular');

%new = load('samples/test_new.mat');
%P = imread ('samples/C0009237.jpg');

P = double (P);
P = imresize(P,[1024,1024]);
%disp(max(P));

padded_image = padarray(P, [4, 4], 0, 'both');
%S = imread('samples/Motion Free Sinogram.jpg');
%P = translateImage (P,20);
%cube = new.Voxel;
%cube_1 = new.Voxel;
%cube = permute(cube,[3,1,2]);
%cube_1 = permute(cube_1,[3,1,2]);
%angles = linspace2(0, 2*pi,195);
%P = cube(:,:,1);
angles = linspace2(-pi/4, 7*pi/4, 1800); %previously 0, 2*pi, 900
%angles_cor = calculate_theta_prime (angles);
angles_cor1 = angles + randomValues;
%angles2 = linspace2(-pi/4, 7*pi/4, 900);
%angles = linspace2(0, 2*pi, 900);
%proj_geom = astra_create_proj_geom('parallel3d', 1.0, 1.0, 400, 400, angles);
%proj_geom_og = astra_create_proj_geom('parallel3d', 1.0, 1.0, 400, 400, angles);

%proj_geom = astra_create_proj_geom('fanflat', 1.0, 400, angles, 401, 205);
vol_geom = astra_create_vol_geom(1024, 1024); %Previously 512 512
proj_geom = astra_create_proj_geom('parallel', 1.0, 1024, angles);

vol_geom3 = astra_create_vol_geom(1024,1024); %previosuly 512 512
proj_geom3 = astra_create_proj_geom('parallel', 1.0, 1024, angles_cor1); %Number of detectors increase with higher pixels

vol_geom5 = astra_create_vol_geom(1032,1032); %previosuly 512 512
proj_geom5 = astra_create_proj_geom('parallel', 1.0, 1032, angles); %Number of detectors increase with higher pixels
%%
% As before, create a sinogram from a phantom
%P = phantom(256);
%%Introducing a rotation
%P_1 = imrotate( P , 12 );
%P_1 = imresize(P_1,[256,256]);
%figure(1); imshow(P_1, []);
%%
[sinogram_id, sinogram] = astra_create_sino_gpu(P, proj_geom, vol_geom);
%[sinogram_id_og, sinogram_og] = astra_create_sino_gpu(P, proj_geom2, vol_geom2);
[sinogram_id_og1, sinogram_og1] = astra_create_sino_gpu(P, proj_geom3, vol_geom3);

[sinogram_id_tr, sinogram_tr] = astra_create_sino_gpu(padded_image, proj_geom5, vol_geom5);
%figure(1); imshow(P, []);
%sinogram = motionBlur(sinogram, 15);
%sinogram_og = translateImage(sinogram_og,3);
%figure(2); imshow(sinogram, []);
%figure(3); imshow(sinogram_og, []);
%sinogram (200:300,:) = sinogram_og (200:300,:);
%figure, imshow(sinogram, []);

A =  randi([1, 1600]);
%B = randi([1, 900],1,800);
B = A + 200; %200
%for i =200 1:length(A)
%sinogram (A(i), :) = (1/sqrt(3)* cos (pi/6)*cos (pi/6))*sinogram_og(B(i),:);
%end
%for i = 300:320
%sinogram (i, :) = sinogram_og (i,:);
%end
for i = A:B
sinogram (i, :) = sinogram_og1 (i,:);
end

%%
C = randi([1, 1600]);
D = C + 200;
shift_amt = 5;

for i = C:D
    row = sinogram(i, :);

    % Interpolate the values to shift right by 5 pixels
    % For example, row(k+5) = row(k)
    shifted_row = zeros(1, length(row));
    for k = shift_amt+1:length(row)
        % Use linear interpolation from the previous values
       shifted_row(k) = row(k - shift_amt);
    end

    % Fill first few pixels by extrapolating or copying edge
    shifted_row(1:shift_amt) = row(1);  % or: interp1(1+shift_amt:length(row), row(1:end-shift_amt), 1:shift_amt, 'linear', 'extrap')

    sinogram(i, :) = shifted_row;
end




%%
%%
recon_2d = sinogram;
end