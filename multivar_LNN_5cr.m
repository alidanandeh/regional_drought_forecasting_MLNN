% multiivariate LNN for 5 Centriod running simultaneously.
clc;
clear all;
fileNames = {"LNN1_Data_multi.xlsx", "LNN2_Data_multi.xlsx", "LNN3_Data_multi.xlsx", "LNN4_Data_multi.xlsx", "LNN5_Data_multi.xlsx"};
numFiles = length(fileNames);

%% Initialize results storage
allResults = cell(numFiles, 1);

%% Create a single figure for all plots
figure('Name', 'All LNN Results', 'Position', [100, 100, 1400, 1000]);

%% Process each file
for fileIdx = 1:numFiles
    currentFile = fileNames{fileIdx};
    fprintf('\n=== Processing file %d of %d: %s ===\n', fileIdx, numFiles, currentFile);
    
    %% Load Data
    try
        data = readmatrix(currentFile);
        fprintf('Successfully loaded data from %s\n', currentFile);
    catch
        fprintf('Error: Could not load file %s. Skipping...\n', currentFile);
        continue;
    end
    
    X = data(:,1:10)'; % Inputs: 10 predictors, shape = [10, numSamples]
    Y = data(:,11)'; % Target SPEI series, shape = [1, numSamples]
    numSamples = size(X,2);

    % Create date sequence (monthly starting from April 1952)
    startDate = datetime(1952,4,1);
    dates = startDate + calmonths(0:numSamples-1);

    splitIdx = floor(0.8*numSamples);
    XTrain = X(:,1:splitIdx);
    YTrain = Y(:,1:splitIdx);
    XTest = X(:,splitIdx+1:end);
    YTest = Y(:,splitIdx+1:end);

    % Split dates accordingly
    datesTrain = dates(1:splitIdx);
    datesTest = dates(splitIdx+1:end);

    %% Define Custom Liquid Layer
    numInputs = size(XTrain,1);
    numHidden = 16; % number of liquid units (tunable)
    layers = [
        featureInputLayer(numInputs,"Name","input")
        LiquidLayer(numHidden,numInputs,"liquid1")
        fullyConnectedLayer(1,"Name","fc")
        regressionLayer("Name","regression")
    ];

    %% Training Options
    options = trainingOptions("adam", ...
        "MaxEpochs",100, ...
        "MiniBatchSize",32, ...
        "InitialLearnRate",0.01, ...
        "Plots","none", ...
        "Verbose",false);

    %% Train Model
    fprintf('Training model for %s...\n', currentFile);
    net = trainNetwork(XTrain',YTrain',layers,options);

    %% Predict
    YTrainPred = predict(net,XTrain');
    YTestPred = predict(net,XTest');

    %% Performance Metrics
    % Functions for metrics
    MAE = @(y,yhat) mean(abs(y - yhat));
    MSE = @(y,yhat) mean((y - yhat).^2);
    RMSE = @(y,yhat) sqrt(mean((y - yhat).^2));
    NSE = @(y,yhat) 1 - sum((y - yhat).^2) / sum((y - mean(y)).^2);

    % Training metrics
    mae_train = MAE(YTrain',YTrainPred);
    mse_train = MSE(YTrain',YTrainPred);
    rmse_train = RMSE(YTrain',YTrainPred);
    nse_train = NSE(YTrain',YTrainPred);

    % Testing metrics
    mae_test = MAE(YTest',YTestPred);
    mse_test = MSE(YTest',YTestPred);
    rmse_test = RMSE(YTest',YTestPred);
    nse_test = NSE(YTest',YTestPred);

    %% Report Results
    fprintf("\nPerformance Metrics for %s:\n", currentFile);
    fprintf("Training -> MAE: %.4f, MSE: %.4f, RMSE: %.4f, NSE: %.4f\n", mae_train, mse_train, rmse_train, nse_train);
    fprintf("Testing  -> MAE: %.4f, MSE: %.4f, RMSE: %.4f, NSE: %.4f\n", mae_test, mse_test, rmse_test, nse_test);

    %% Export Results to Excel for this file
    % Create performance metrics table
    metrics = {
        'MAE', mae_train, mae_test;
        'MSE', mse_train, mse_test;
        'RMSE', rmse_train, rmse_test;
        'NSE', nse_train, nse_test
    };
    metricsTable = cell2table(metrics, 'VariableNames', {'Metric', 'Training', 'Testing'});

    % Create prediction results table with dates
    allPredictions = [YTrainPred; YTestPred];
    setLabels = [repmat({'Training'}, length(YTrainPred), 1); repmat({'Testing'}, length(YTestPred), 1)];

    resultsTable = table(dates', Y', allPredictions, setLabels, ...
        'VariableNames', {'Date', 'Actual_SPEI', 'Predicted_SPEI', 'Set'});

    % Write to Excel file for this dataset
    [~, baseFileName, ~] = fileparts(currentFile);
    filename = baseFileName + "_results.xlsx";
    writetable(metricsTable, filename, 'Sheet', 'Performance_Metrics');
    writetable(resultsTable, filename, 'Sheet', 'Prediction_Results');

    fprintf('Results exported to: %s\n', filename);

    %% Store results for summary
    allResults{fileIdx} = struct(...
        'FileName', currentFile, ...
        'TrainingMetrics', [mae_train, mse_train, rmse_train, nse_train], ...
        'TestingMetrics', [mae_test, mse_test, rmse_test, nse_test], ...
        'Predictions', resultsTable, ...
        'Network', net ...
    );

    %% Create subplot for this file in the main figure
    subplot(3, 2, fileIdx); % 3 rows, 2 columns layout (for 5 plots + 1 empty or legend)
    
    plot(dates, Y, '-', 'DisplayName', 'Actual', 'Color', 'blue', 'LineWidth', 1);
    hold on;
    plot(datesTrain, YTrainPred, '-', 'DisplayName', 'Train Fitted', 'Color', 'red', 'LineWidth', 1.5);
    plot(datesTest, YTestPred, '-', 'DisplayName', 'Test Forcast', 'Color', [1, 0.4, 0.6], 'LineWidth', 1.5); % Pink color

    % Add train/test split line (black, no label, excluded from legend)
    xline(datesTrain(end), '--k', 'LineWidth', 1.5); % Black dashed line, no label

    legend('Location', 'best', 'FontSize', 12,'Orientation', 'horizontal');
    xlabel('Date', 'FontSize', 14);
    ylabel('SPEI Value', 'FontSize', 14);
    title(sprintf('%s', baseFileName), 'FontSize', 12, 'Interpreter', 'none');
    grid on;
    set(gca, 'FontSize', 10);
end

%% Add a summary subplot or adjust layout
% Adjust the last subplot position if needed, or add a legend/description
if numFiles == 5
    % Create a common legend in the last subplot position
    subplot(3, 2, 6);
    axis off; % Turn off axes for the legend box
    
    % Create dummy plots for legend
    hold on;
    plot(NaN, NaN, '-b', 'LineWidth', 2, 'DisplayName', 'True Series');
    plot(NaN, NaN, '-r', 'LineWidth', 2, 'DisplayName', 'Training Predicted');
    plot(NaN, NaN, 'Color', [1, 0.4, 0.6], 'LineWidth', 2, 'DisplayName', 'Testing Predicted');
    plot(NaN, NaN, '--k', 'LineWidth', 2, 'DisplayName', 'Train/Test Split');
    
    legend('Location', 'bestoutside', 'FontSize', 12,'Orientation', 'horizontal');
    title('Legend', 'FontSize', 14);
end

% Add a overall title for the figure
sgtitle('LNN Model Predictions for All Datasets', 'FontSize', 16, 'FontWeight', 'bold');

%% Create summary report across all files
fprintf('\n=== SUMMARY REPORT ACROSS ALL FILES ===\n');

% Create summary table
summaryData = cell(numFiles, 9);
for fileIdx = 1:numFiles
    if ~isempty(allResults{fileIdx})
        results = allResults{fileIdx};
        summaryData{fileIdx, 1} = results.FileName;
        summaryData{fileIdx, 2} = results.TrainingMetrics(1); % MAE Train
        summaryData{fileIdx, 3} = results.TrainingMetrics(2); % MSE Train
        summaryData{fileIdx, 4} = results.TrainingMetrics(3); % RMSE Train
        summaryData{fileIdx, 5} = results.TrainingMetrics(4); % NSE Train
        summaryData{fileIdx, 6} = results.TestingMetrics(1);  % MAE Test
        summaryData{fileIdx, 7} = results.TestingMetrics(2);  % MSE Test
        summaryData{fileIdx, 8} = results.TestingMetrics(3);  % RMSE Test
        summaryData{fileIdx, 9} = results.TestingMetrics(4);  % NSE Test
    end
end

summaryTable = cell2table(summaryData, 'VariableNames', {...
    'FileName', 'MAE_Train', 'MSE_Train', 'RMSE_Train', 'NSE_Train', ...
    'MAE_Test', 'MSE_Test', 'RMSE_Test', 'NSE_Test'});

% Display summary
disp(summaryTable);

% Export summary to Excel
writetable(summaryTable, 'LNN_All_Results_Summary.xlsx');
fprintf('Summary report exported to: MultiLNN_All_Results_Summary.xlsx\n');

fprintf('\n=== PROCESSING COMPLETE ===\n');
fprintf('All files have been processed. Individual results saved as "*_results.xlsx"\n');
fprintf('All plots are displayed in a single figure for comparison.\n');