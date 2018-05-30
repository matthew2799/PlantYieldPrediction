function showMultIm(array, titleStr)

hFig = figure;
% set(hFig, 'Position', [0,40, 1280,960]);

for i = 1:20
    subplot(4,5,i)
    imshow(array{i})
end

mtit(hFig, titleStr);

end