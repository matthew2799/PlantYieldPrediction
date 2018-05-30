imtool close all;
close all; 
clear;

display = 1;

addpath('../helpers/');

image = imread('../../../images_and_annotations/PSI_Tray031/p-1/PSI_Tray031_2015-12-29--15-32-47_top_1-1_495.png');

% Get the image mask
mask = get_plant_mask(image, 5000, 0);

hFig = figure;
imshow(image)
title('Raw Image')

iFig = figure;
imshow(mask)
title('')
