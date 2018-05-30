function [Xm, Ym, Zm] = color_index(image, mask, display)


    plant   = im2double(image).*repmat(mask, [1,1,3]);
    cie_xyz = rgb2xyz(plant, 'WhitePoint', 'd50');
   
    if display == 1 
        imtool(cie_xyz)
    end
    

    
    X = cie_xyz(:, :, 1);
    Y = cie_xyz(:, :, 2);
    Z = cie_xyz(:, :, 3);
    
    % min(min(nonzeros(X)))
    % max(max(nonzeros(X)))    
    Xm = mean(mean(nonzeros(X)));
    Ym = mean(mean(nonzeros(Y)));
    Zm = mean(mean(nonzeros(Z)));
   
end