imtool close all;
close all;
clear;

basedir =  '../images_and_annotations/PSI_Tray034/';
imagedir = '../images_and_annotations/PSI_Tray034/tv/';

files = dir(strcat(imagedir,'/*.png'));
numfiles = length(files);
images = cell(1,numfiles);

for i = 1:20
    d = strcat('../images_and_annotations/PSI_Tray034/', strcat('p-',num2str(i)));
    if ~isdir(d)
        mkdir('../images_and_annotations/PSI_Tray034/',  strcat('p-',num2str(i)));
    end
end

for ii=1:numfiles
    name = files(ii).name;
    if name(1) ~= '.'
        all_plants = imread(strcat(imagedir, name));
        % splitandstore(all_plants,name,183,2412,80,1858);
        l = 183; r = 2412; t = 80; b = 1858;
        % cropped borders from image
        cropped = all_plants(t:b,l:r,:);
        [h,w,c] = size(cropped);
        y = 1;
        x = 1;
        w = floor(w/5);
        h = floor(h/4);
        count = 1;
        for i = 1:4
            for j = 1:5
                plant = cropped(y:y+h,x:x+w,:);
                id = strsplit(name,'.');
                id = strcat(id{1},'_',num2str(i),'-',num2str(j),'_', num2str(ii));
                imwrite(plant, strcat(basedir, 'p-', num2str(count), '/', id, '.png'));
                x = x + w - 1;
                count = count + 1;
            end
            x = 1;
            y = y + h;
        end
    end
end


% i = 1;
% for file = files'
%     path = strcat(file.folder, '\', file.name);
%     [folder, name, ext] = fileparts(path);
%     if strcmp(ext, '.png') && name(1) ~= '.'
%         image = imread(strcat(base,'/',name,'.png'));
%     end
% end

