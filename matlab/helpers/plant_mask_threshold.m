function mask = plant_mask_threshold(image, display)
   red   = image(:,:,1);
   green = image(:,:,2);
   blue  = image(:,:,3);

   leaves = green >= 85 & red < 130 & red > 75 & blue < 75;

   se = strel('disk', 5);
   mask = imopen(leaves, se);
   
   if display == 1
       imtool(image)
       imtool(mask)
   end
end