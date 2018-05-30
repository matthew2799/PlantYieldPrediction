close all;
clear;

dataset = readtable('../../data/all_data.csv');
data_array = table2array(dataset);
for i = 31:34
    tray = data_array(data_array(:, 8) == i, :);
    for j = 1:20
        pos = tray(tray(:,9) == j, :);
        
        dry_weight = pos(1,6);
        max_day = max(pos(:,2));
        
        figure;
        subplot(2,1,1)
        plot(pos(:,2), pos(:,3), '.', 'markers', 20)
        hold on;
        plot(max_day,dry_weight,'r.','markers', 20);
        title(sprintf('Predicted Current Yield for Plant: T%dP%d', i, j)) 
        xlabel('Days after Sewing')
        ylabel('Predicted Dry Weight (mg)')
        legend('Predicted Values','Target Value', 'Location', 'northwest')
        
        subplot(2,1,2)
        plot(pos(:,2), pos(:,4), '.', 'markers', 20)
        hold on;
        plot(max_day,dry_weight,'r.','markers',20);
        title(sprintf('Predicted Final Yield for Plant: T%dP%d', i, j)) 
        xlabel('Days after Sewing')
        ylabel('Predicted Dry Weight (mg)')    
        legend('Predicted Values','Target Value', 'Location', 'northwest')
    end
    break;
end