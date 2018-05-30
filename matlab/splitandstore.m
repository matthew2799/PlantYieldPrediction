function splitandstore(image, name, l, r, t, b)
    
    % l = left border
    % r = right border
    % t = top border
    % b = bottom border
    % w = width

    % cropped borders from image
    cropped = image(t:b,l:r,:);
    imtool(cropped);
    [h,w] = size(cropped)
    y = 1;
    x = 1;
    w = floor(w/4)
    h = floor(h/4)
    plants = cell(1,20);
    names  = cell(1,20);
    
    for i =1:4
        for j = 1:5
            plants{i} = cropped(y:y+h,x:x+w,:);
            names{i} = strcat(name,num2str(i),'-',num2str(j));
            x = x + w
        end
        x = 1
        y = y + h
    end
    
    showMultIm(plants, 'All the plants Thanks');
end