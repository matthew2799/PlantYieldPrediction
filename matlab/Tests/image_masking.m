imtool close all;
close all; 
clear

addpath('../helpers/')

image = imread('../../images_and_annotations/PSI_TraY031/p-2/PSI_Tray031_2016-01-09--16-16-24_top_1-2_813.png');
image1 = imread('../../images_and_annotations/PSI_TraY031/p-2/PSI_Tray031_2016-01-01--09-34-15_top_1-2_574.png');

lab_im = rgb2lab(image);

ab = lab_im(:,:,2:3);

[nRows,nCols,nChannels] = size(ab);
ab = reshape(ab, nRows*nCols, 2);

nColors = 3;

[clusters, cluster_center] = kmeans(ab, nColors, 'distance','sqEuclidean', ...
                                           'replicates', 10);

pixel_id = reshape(clusters, nRows, nCols);
% imtool(pixel_id, [])

mean_cluster_value = mean(cluster_center, 2);
[tmp, idx] = sort(mean_cluster_value);
green_cluster_value = idx(3);


green_idx = zeros(size(pixel_id));
green_idx(pixel_id == green_cluster_value) = 1;

figure;
imshow(green_idx);

basic_thresh = plant_mask_threshold(image, 1);