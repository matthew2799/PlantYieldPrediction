function normDistance = leaf_continuity(mask, leaf_area)

    B = bwboundaries(mask);
    
    distance = zeros(1,length(B));
    
    for k = 1:length(B)
        currentDistance = 0;
        edge = B{k};

        for i = 1:length(edge) - 1        
            x1 = edge(i  , :);
            x2 = edge(i+1, :);

            cum_len = ((x1(1,1) - x2(1,1)).^2 + ...
                (x1(1,2) - x2(1,2)).^2).^0.5;
            currentDistance = currentDistance + cum_len;
        end

        x1 = edge(i , :);
        x2 = edge(1 , :);

        cum_len = ((x1(1,1) - x2(1,1)).^2 + (x1(1,2) - x2(1,2)).^2).^0.5;
        currentDistance = currentDistance + cum_len;   

        distance(k) = currentDistance;
    end

    totalDistance = sum(distance);
    normDistance  = totalDistance / leaf_area;
end