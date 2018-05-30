close all;
clear;

dataset = readtable('../../data/all_data.csv');
data_array = table2array(dataset);

error = cell(1,20)
for i = 31:34
    tray = data_array(data_array(:, 8) == i, :);
    for j = 1:20
        pos = tray(tray(:,9) == j, :);
        
        div = pos(:,4) ./ pos(:,6);
        
        error = ones(size(pos(:,2)));
        figure;
        semilogy(pos(:,2), div(:), '.', 'markers', 20)
        hold on;
        plot(pos(:,2), error(:), 'r-')
        title(sprintf('Future Yield Prediction Error: T%dP%d', i, j)) 
        legend('Accuracy','Target Accuracy', 'Location', 'northeast')
        xlabel('Days after Sewing')
        ylabel('Prediction Error (log(pred/target))')
    end
    break;
end