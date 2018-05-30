imtool close all;
close all; 
clear;

display = 1;

addpath('../annotations/');

image = imread('../../images_and_annotations/PSI_Tray031/p-1/PSI_Tray031_2016-01-03--20-21-58_top_1-1_620.png');

% Get the image mask
mask = get_plant_mask(image, 5000, 0);
area = sum(sum(mask));

% area = sum(sum(mask));
leaves = im2double(image).*repmat(mask, [1,1,3]);

% get the image as a skeleton

[B, L] = bwboundaries(mask);
