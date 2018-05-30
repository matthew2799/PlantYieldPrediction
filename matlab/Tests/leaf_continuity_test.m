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

imshow(label2rgb(L, @jet, [.5 .5 .5]))
hold on

distance = zeros(1,length(B));
for k = 1:length(B)
    currentDistance = 0;
    edge = B{k};
    
    for i = 1:length(edge) - 1        
        x1 = edge(i  , :);
        x2 = edge(i+1, :);
        
        cum_len = ((x1(1,1) - x2(1,1)).^2 + (x1(1,2) - x2(1,2)).^2).^0.5;
        currentDistance = currentDistance + cum_len;
    end
    
    x1 = edge(i , :);
    x2 = edge(1 , :);

    cum_len = ((x1(1,1) - x2(1,1)).^2 + (x1(1,2) - x2(1,2)).^2).^0.5;
    currentDistance = currentDistance + cum_len;   
    
    distance(k) = currentDistance;
end

totalDistance = sum(distance);
normDistance = totalDistance/area;

