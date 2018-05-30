imtool close all;
close all; 
clear;

% Debug enable
display_en = 0;

addpath('./helpers/');
addpath('./annotations/');

for set = 31:34
    base = sprintf('../images_and_annotations/PSI_Tray0%d/p-', set);
    test_set = [1, 50, 100, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 900, 920, 1000, 1020, 1100];

    sample_interval = hours(4);

    for pot = 1:20

       folder = strcat(base, num2str(pot), '/');
       % PSI_Tray031_2015-12-14--12-54-06_top_1-1-4
       files = dir(strcat(folder,'*.png'));
       numfiles = length(files);

%        fid = fopen(strcat(folder, sprintf('PSI_Tray0%dp-%d.csv', set, pot)), 'w');

       % Print header line 
%        fprintf(fid, 'name,datetime,image_num,leaf_area,cont,Xm,Ym,Zm\n');
       % annotations = sprintf('%d,%d,%d,%d,%d', la, continuity, Xm, Ym, Zm);
       % fprintf(fid, ',%d,%s,%s,%s\n', name, strcat(date, ':', time), image_num, annotations);

       % Init variables
       la = 0;
       last_area = 0;
       for image_num = 1:numfiles
           fprintf('Tray: %d, Plant: %d, Image: %d \n', set, pot, image_num)
           name = files(image_num).name;
           tokens = strsplit(name, '_');
           tray_box = tokens{2};
           raw_datetime = tokens{3};
           label = tokens{length(tokens)};

           % separate the date and time
           tokens = strsplit(raw_datetime, '--');
           date = tokens{1};
           time = tokens{2};

           % Check if datetime is past the required interval
           dt = datetime(raw_datetime, 'InputFormat', 'yyyy-MM-dd--HH-mm-ss');

           if image_num == 1
               last_sample = dt;
           else
               if dt - last_sample > sample_interval
    %                 display(dt - last_sample)
                    last_sample = dt;
               else
                   continue;
               end
           end

           % Remove this line once file names have been fixed
           tokens = strsplit(label, '.');
           order = tokens{1};

           image = imread(strcat(folder,name));


           basic_thresh = plant_mask_threshold(image, display_en);
           basic_area = leaf_area(basic_thresh, display_en);

           if 0.5 * last_area > basic_area
               disp('Plant harvested, breaking loop...')
               if display_en == 1
                   figure;
                   imshow(image)
                   title('Detected Plant Harvest')
               end
               break
           end

           % Get the most accurate mask 
           if la <= 5000
               mask = basic_thresh;
               la = basic_area;
           elseif la <= 55000
               mask = plant_mask_kmeans(image, 3, display_en);       
               la   = leaf_area(mask, display_en);
           else
               mask = plant_mask_kmeans(image, 2, display_en);
               la   = leaf_area(mask, display_en);
           end

           % Get plant annotations

           continuity = leaf_continuity(mask, la);
           [Xm, Ym, Zm] = color_index(image, mask, display_en);      

           % pack the data into rows to append to csv file
           row = strcat(order,',',raw_datetime,',',num2str(la));

           % tray, plant, image_name, annotations
           annotations = sprintf('%d,%d,%d,%d,%d', la, continuity, Xm, Ym, Zm);
%            fprintf(fid, '%s,%s,%d,%s\n', name, strcat(date, ':', time), image_num, annotations);

           last_area = la;
       end
    end
end