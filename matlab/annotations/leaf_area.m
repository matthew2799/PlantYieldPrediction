function area = leaf_area(mask, display)
   area = sum(sum(mask));
   if display == 1
       fprintf('Leaf Area: %d \n', area)
   end
end