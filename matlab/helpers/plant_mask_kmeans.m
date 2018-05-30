function [mask] = plant_mask_kmeans(image, nClusters, display)
    
    lab_im = rgb2lab(image);
    
    ab = lab_im(:,:,2:3);
    
    [nRows, nCols, nChannels] = size(ab);
    ab = reshape(ab, nRows*nCols, 2);

    [clusters, cluster_centers] = kmeans(ab, nClusters, 'distance', ...
        'sqEuclidean', 'replicates', 10);
                                       
    cluster_map = reshape(clusters, nRows, nCols);
    
    mean_cluster_value = mean(cluster_centers, 2);
    [tmp, idx] = sort(mean_cluster_value);
    green_cluster_value = idx(nClusters); % Found through experiment
    
    mask = zeros( size( cluster_map ));
    mask(cluster_map == green_cluster_value) = 1;
    
    if display == 1
        imtool(mask)
    end
    
    % Simple binary opening to clear up white fuzz
    se = strel('disk', 1);
    mask = imclose(mask, se);
    mask = imopen(mask, se);
    
    if display == 1
        figure;
        imshow(mask)
    end
end