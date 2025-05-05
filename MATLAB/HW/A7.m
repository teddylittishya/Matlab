% Load the handwritten digits dataset (assuming you have it)
% load("usps_all.mat");
filePath = '/Users/teddythomas/Documents/MATLAB/usps_all.mat';
loadedData = load(filePath);

% Split indices using indexing
train_indices = 1:800;
val_indices = 801:1000;
test_indices = 1001:1100;

% Create subsets
train_set = data(:, train_indices, :);
val_set = data(:, val_indices, :);
test_set = data(:, test_indices, :);


% Initialize labels for training and validation sets
train_labels = [];
val_labels = [];

for digit = 1:10
    train_labels =[train_labels, digit * ones(1, 800)];
    val_labels =[val_labels, digit * ones(1, 200)];
    
end
%train_labels = repelem(0:9, 800)';
% val_labels = repelem(0:9, 200)';

% Initialize variables to store validation errors
k_values = 1:20;
validation_errors = zeros(size(k_values));

% Train kNN models for different k values
for k = k_values
    mdl = fitcknn(train_set, train_labels, 'NumNeighbors', k, 'Distance', 'euclidean');
    predictions = predict(mdl, val_data);
    validation_errors(k) = sum(predictions ~= val_labels) / numel(val_labels);
end

% Find the best k
[min_error, best_k] = min(validation_errors);
disp(['Best k: ', num2str(best_k)]);

% Test accuracy on the testing set using the best k
mdl_best_k = fitcknn(train_data, train_labels, 'NumNeighbors', best_k, 'Distance', 'euclidean');
test_predictions = predict(mdl_best_k, test_data);
test_error_rate = sum(test_predictions ~= test_labels) / numel(test_labels);
disp(['Testing error rate: ', num2str(test_error_rate)]);
