% Load USPS dataset
load("usps_all.mat");

% Select the classes corresponding to the last three digits of your UMD ID
class_labels = [9, 3, 8];  % Corresponding to the last three digits of your UMD ID

% Initialize variables to store training and testing data
train_data = [];
test_data = [];
train_labels = [];
test_labels = [];

% Reshape each image to 16x16
data = reshape(data, 16, 16, 1100, 10);

% Iterate over selected classes
for i = 1:numel(class_labels)
    % Select instances for the current class
    class_idx = class_labels(i) + 1; % Adjust for 0-based indexing
    class_data = data(:, :, :, class_idx);
    
    % Split the instances into training and testing data
    train_data = cat(3, train_data, class_data(:, :, 1:1000));
    train_labels = [train_labels; repmat(class_labels(i), 1000, 1)];
    
    test_data = cat(3, test_data, class_data(:, :, 1001:1100));
    test_labels = [test_labels; repmat(class_labels(i), 100, 1)];
end

% Reshape data for SVM training
train_data = reshape(train_data, size(train_data, 1) * size(train_data, 2), size(train_data, 3))';
test_data = reshape(test_data, size(test_data, 1) * size(test_data, 2), size(test_data, 3))';

% Normalize pixel values
train_data = double(train_data) ./ 255;
test_data = double(test_data) ./ 255;

% Train a multi-class classifier using one-vs-all SVM
SVMModel = fitcecoc(train_data, train_labels, 'Learners', 'svm');

% Predict labels for test data
predictedLabels = predict(SVMModel, test_data);

% Calculate accuracy
accuracy = sum(predictedLabels == test_labels) / numel(test_labels);
disp(['Accuracy: ', num2str(accuracy)]);

%%
% Predict labels for the remaining 300 pictures using the trained SVM model
predictedLabels_remaining = predict(SVMModel, test_data);

% Calculate global accuracy
global_accuracy = sum(predictedLabels_remaining == test_labels) / numel(test_labels);

% Calculate local accuracy (accuracy by label)
local_accuracy = zeros(1, numel(class_labels));
for i = 1:numel(class_labels)
    % Extract true labels and predicted labels for the current class
    true_labels_class = test_labels(test_labels == class_labels(i));
    predicted_labels_class = predictedLabels_remaining(test_labels == class_labels(i));
    
    % Calculate accuracy for the current class
    local_accuracy(i) = sum(predicted_labels_class == true_labels_class) / numel(true_labels_class);
end

% Display global accuracy
disp(['Global Accuracy: ', num2str(global_accuracy)]);

% Display local accuracy
disp('Local Accuracy (Accuracy by Label):');
for i = 1:numel(class_labels)
    disp(['Class ', num2str(class_labels(i)), ' Accuracy: ', num2str(local_accuracy(i))]);
end

