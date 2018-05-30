function [mask] = get_plant_mask(image, last_area, display)
   
   if last_area <= 2000
       mask = plant_mask_threshold(image, display);
   elseif last_area <= 55000
       mask = plant_mask_kmeans(image, 3, display);
   else
       mask = plant_mask_kmeans(image, 2, display);
   end
end