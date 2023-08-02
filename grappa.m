%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Project       : Parallel Imaging - GRAPPA
% Author        : Hakan Gulec
% Supervisor(s) : Berkin Bilgic & Yohan Jun
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;
clc;
close all

addpath utils\

%-------------------------------------------------------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Description %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%-------------------------------------------------------------------------%

% This project implements GRAPPA(Generalized Autocalibrating Partially 
% Parallel Acquisitions) parallel imaging technique for MRI image 
% reconstruction. The data used in this code is provided as a .mat file. 

%-------------------------------------------------------------------------%


%% Load data


load data\sim_8ch_data.mat


%% Get the original image, coil sensitivity maps and coil images


% Ground truth image 
image_original = anatomy_orig / max(abs(anatomy_orig(:)));

% Coil sensitivity maps
coil_sensitivity_maps = b1 / max(abs(b1(:)));

% Number of coils, 8 for this data
number_of_coils = size(coil_sensitivity_maps, 3);

% MR images under each coil
image_coils = repmat(image_original, [1, 1, number_of_coils]) .* coil_sensitivity_maps;

% I used the mosaic function which was sent in the zip file
mosaic(image_original       , 1, 1, 1, 'Original Image', [0  1], 0);
mosaic(coil_sensitivity_maps, 2, 4, 2, 'Coil Maps'     , [0 .5], 0);
mosaic(image_coils          , 2, 4, 3, 'Coil Images'   , [0 .5], 0);


%% Get the k-space data by taking Fourier Transform of coil images


% Define k-space of the coil images variable 
kspace_coils = zeros(size(image_coils));

% Apply Fourier Transform to obtain k-space of images under each coil 
for i = 1:number_of_coils

    kspace_coils(:, :, i) = fftshift(fftn(fftshift(image_coils(:, :, i))));

end

mosaic(kspace_coils, 2, 4, 4, 'K-Space of Coils', [0 1], 0);


%% Define the variables for GRAPPA weight set


% Get ACS_size phase-encoding lines as fully-sampled and let the kernel
% size be 3*2
ACS_size    = 32                    ;
kernel_size = 3                     ;
N_R         = size(kspace_coils, 2) ;
N_PE        = ACS_size              ;
s_R         = 3                     ;
s_PE        = 2                     ;

% Define the acceleration factors, I wrote this code only for R = 2, need
% some generalization
R_x = 2;
R_y = 2;


%% Get the ACS data from the coil k-spaces


% ACS test data for GRAPPA weight set
ACS_data = kspace_coils(((end - ACS_size) / 2 + 1):((end + ACS_size) / 2), :, :);

% Display of the ACS data
mosaic(ACS_data, 2, 4, 5, 'ACS Data', [0 1], 0);

% Compute the number of repetitions
number_of_repetitions = (N_R - s_R + 1) * (N_PE - R_y * (s_PE - 1));

% Define the empty source and target matrices
S_ACS = zeros(number_of_repetitions, 2 * kernel_size * number_of_coils)        ;
T_ACS = zeros(number_of_repetitions, ((kernel_size - 1) * number_of_coils / 2));


%% Compute the source and target ACS matrices


count = 1;

for i = 2:(ACS_size - 1)

    for j = 2:(N_R - 1)
    
        S_temp1            = ACS_data(i - 1, (j - 1):(j + 1), :); 
        S_temp2            = ACS_data(i + 1, (j - 1):(j + 1), :);
        S_temp             = [S_temp1 S_temp2]                  ;
        S_temp             = reshape(S_temp, 1, [])             ;

        S_ACS(count, :, :) = S_temp;
        
        T_ACS(count, :) = ACS_data(i, j, :);
        
        count = count + 1;

    end

end


%% Find the GRAPPA weight set


% Solve the equation W = S_ACS_inverse*T_ACS
grappa_weight_set = S_ACS \ T_ACS;


%% Create variables for GRAPPA reconstruction


% Create undersampled k-space
kspace_undersampled = zeros(size(kspace_coils));

kspace_undersampled(((end - ACS_size) / 2 + 1):((end + ACS_size) / 2), :, :) = ...
    kspace_coils(((end - ACS_size) / 2 + 1):((end + ACS_size) / 2), :, :);

kspace_undersampled(1:R_y:((end - ACS_size) / 2 - 1), :, :)     = kspace_coils(1:R_y:((end - ACS_size) / 2 - 1), :, :)  ;
kspace_undersampled(((end + ACS_size) / 2 + 2):R_y:end, :, :)   = kspace_coils(((end + ACS_size) / 2 + 2):R_y:end, :, :);

% Define aliased images variable 
image_aliased = zeros(size(image_coils));

% Create aliased image by taking Fourier transform
for i = 1:number_of_coils

    image_aliased(:, :, i) = fftshift(ifftn(fftshift(kspace_undersampled(:, :, i))));

end

% Display of the ACS data
mosaic(image_aliased, 2, 4, 6, 'Aliased Images', [0 .5], 0);


%% Implement GRAPPA reconstruction


% GRAPPA for the part above the ACS lines
for i = 2:R_y:((N_R - ACS_size) / 2)
    
    for j = 2:(N_R - 1)
    
        S_temp1 = kspace_undersampled(i - 1, (j - 1):(j + 1), :); 
        S_temp2 = kspace_undersampled(i + 1, (j - 1):(j + 1), :);
        S_temp = [S_temp1 S_temp2];
        S_temp = reshape(S_temp, 1, []);
        
        kspace_undersampled(i, j, :) = S_temp * grappa_weight_set;

    end

end

% GRAPPA for the part below the ACS lines
for i = ((N_R + ACS_size) / 2 + 1):R_y:(N_R - 1)
    
    for j = 2:(N_R - 1)
    
        S_temp1 = kspace_undersampled(i - 1, (j - 1):(j + 1), :); 
        S_temp2 = kspace_undersampled(i + 1, (j - 1):(j + 1), :);
        S_temp = [S_temp1 S_temp2];
        S_temp = reshape(S_temp, 1, []);
        
        kspace_undersampled(i, j, :) = S_temp * grappa_weight_set;

    end

end


% Display of the reconstructed k-space
mosaic(kspace_undersampled, 2, 4, 7, 'Reconstructed k-space', [0 1], 0);


%% Reconstruct the image


% Define the empty reconstructed image variable 
image_reconstructed = zeros(size(kspace_undersampled));

% Apply inverse Fourier transform to obtain the images under reconstructed
% k-spaces
for i = 1:number_of_coils

    image_reconstructed(:, :, i) = fftshift(ifftn(fftshift(kspace_undersampled(:, :, i))));

end

% Compute RMSE and display the reconstructed image
image_reconstructed_average = sqrt(sum(abs(image_reconstructed) .^ 2, 3));

mosaic(image_reconstructed_average, 1, 1, 8, 'Reconstructed Image', [0, .5])

image_original_average = sqrt(sum(abs(image_coils) .^ 2, 3));

rmse_grappa = 100 * norm(image_reconstructed_average(:) - image_original_average(:)) / norm(image_original_average(:));

mosaic(image_original_average - image_reconstructed_average, 1, 1, 9, ['Error: ', num2str(rmse_grappa), ' % RMSE'], [0,.5])
