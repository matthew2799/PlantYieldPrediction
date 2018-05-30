imtool close all;
close all; 
clear;

display = 1;

addpath('../helpers/');

image = imread('../../images_and_annotations/PSI_Tray031/p-1/PSI_Tray031_2016-01-03--20-21-58_top_1-1_620.png');

% Get the image mask
mask = get_plant_mask(image, 5000, 0);

plant = im2double(image).*repmat(mask, [1,1,3]);

R = plant(:, :, 1);
G = plant(:, :, 2);
B = plant(:, :, 3);

%        [ 0.412 0.358 0.180 ]
%   SM = [ 0.213 0.715 0.072 ]
% %      [ 0.019 0.119 0.950 ]
% 
% ID = cat(3, ...
%     0.412*R + 0.587*G + 0.144*B, ...
%     0.213*R + 0.715*G + 0.072*B, ...
%     0.019*R + 0.119*G + 0.950*B);

XYZ = rgb2xyz(plant,'WhitePoint','d50');

X = XYZ(:, :, 1);
Y = XYZ(:, :, 2);
Z = XYZ(:, :, 3);

figure;
imshow(XYZ)

% x = X ./ (X + Y + Z);
% y = Y ./ (X + Y + Z);
% z = Z ./ (X + Y + Z);

xyz_derived = cat(3, x, y, z);

figure;
imshow(XYZ)

figure;
imshow(xyz_derived)

% min(min(nonzeros(X)))
% max(max(nonzeros(X)))
mean(mean(nonzeros(X)))






