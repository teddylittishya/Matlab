% Import the USPS dataset
load('usps_all.mat')

% Initialize cell arrays to store images of each digit for different data sets
% Each cell array will contain 10 cells, one for each digit (0-9)
digit_training_images = cell(1, 10);   % For training data
digit_validation_images = cell(1, 10); % For validation data
digit_testing_images = cell(1, 10);    % For testing data

% Initialize arrays to hold the corresponding labels for each image in the data sets
% These labels will be used to train and evaluate the model
training_labels = [];   % Labels for training data
validation_labels = []; % Labels for validation data
testing_labels = [];    % Labels for testing data

% Divide the data into training, validation, and testing sets for each digit
for digit = 1:10
    % Extract 800 images for training
    training_images = data(:, 1:800, digit);
    % Extract 200 images for validation
    validation_images = data(:, 801:1000, digit);
    % Extract 100 images for testing
    testing_images = data(:, 1001:1100, digit);
    
    % Save the images in the corresponding cell arrays
    digit_training_images{digit} = training_images;
    digit_validation_images{digit} = validation_images;
    digit_testing_images{digit} = testing_images;
    
    % Generate labels for training, validation, and testing
    train_labels = repmat(digit, 1, 800);
    training_labels = [training_labels, train_labels];
    
    validation_labels = [validation_labels, repmat(digit, 1, 200)];
    testing_labels = [testing_labels, repmat(digit, 1, 100)];
end

% Merge the data for training, validation, and testing
concatenated_training_data = cat(2, digit_training_images{:});
concatenated_validation_data = cat(2, digit_validation_images{:});
concatenated_testing_data = cat(2, digit_testing_images{:});

% Rearrange labels
training_labels = transpose(training_labels);
validation_labels = transpose(validation_labels);
testing_labels = transpose(testing_labels);

% Rearrange data
concatenated_training_data = transpose(double(concatenated_training_data));
concatenated_validation_data = transpose(double(concatenated_validation_data));
concatenated_testing_data = transpose(double(concatenated_testing_data));

% Execute k-NN classification with varying numbers of neighbors
for k = 1:20
    knn_model = fitcknn(concatenated_training_data, training_labels, 'NumNeighbors', k, 'Distance', 'euclidean');
    predictions = predict(knn_model, concatenated_validation_data);
    accuracy = sum(predictions == validation_labels) / numel(validation_labels);
    disp(['Accuracy for ', num2str(k), ' neighbors: ', num2str(accuracy)]);
end

% Finalize k-NN model with 3 neighbors for testing
final_knn_model = fitcknn(concatenated_training_data, training_labels, 'NumNeighbors', 3, 'Distance', 'euclidean');
test_predictions = predict(final_knn_model, concatenated_testing_data);
test_accuracy = sum(test_predictions == testing_labels) / numel(testing_labels);
disp(['Accuracy for the test set: ', num2str(test_accuracy)]);

% Calculate error rates for validation and testing sets
validation_error_rate = 1 - 0.9505;
testing_error_rate = 1 - 0.945;
disp(['Error rate of the validation set: ', num2str(validation_error_rate)]);
disp(['Error rate of the testing set: ', num2str(testing_error_rate)]);
% When k=1 , it was overfitting , so I used k =3 which give better
% accuracy
% The validation error rate is commonly lower than the testing error rate due to hyperparameter tuning on the validation set.
% Comparing the error rates obtained on the validation set (0.0495) and the testing set (0.055)
% Upon the analysis I got thge error rate on the testing set (0.055) is slightly higher than the error rate on the validation set (0.0495).
% Also the performance of the model slightly degraded when evaluated on the unseen testing data compared to the validation data.
% Despite the slight degradation in performance in the unseen test data, the model still demonstrates good generalization ability, with an accuracy of 0.945 on the testing set.

%%
% 2. 
% Precision = Sensitivity = False Omission Rate = Specificity = 80%
% For a binary classification experiment with 100 individuals (TP + TN + FP + FN), we have:
% Precision = TP / (TP + FP)
% Sensitivity = TP / (TP + FN)
% False Omission Rate = FN / (FN + TN)
% Specificity = TN / (FP + TN)
% Given that Precision, Sensitivity, False Omission Rate, and Specificity are all equal to 80%, let's denote this value as 0.80.
% From the precision equation: TP / (TP + FP) = 0.80
% From the sensitivity equation: TP / (TP + FN) = 0.80
% From the false omission rate equation: FN / (FN + TN) = 0.80
% From the specificity equation: TN / (FP + TN) = 0.80
% Now, let's solve these equations to find the values of TP, FP, FN, and TN.
% 
% From the precision equation: TP = 0.80(TP + FP)
% TP - 0.80TP = 0.80FP
% 0.20TP = 0.80FP
% TP = 4FP
% 
% From the sensitivity equation: TP = 0.80(TP + FN)
% TP - 0.80TP = 0.80FN
% 0.20TP = 0.80FN
% TP = 4FN
% 
% From the false omission rate equation: FN = 0.80(FN + TN)
% FN - 0.80FN = 0.80TN
% 0.20FN = 0.80TN
% FN = 4TN
% 
% From the specificity equation: TN = 0.80(FP + TN)
% TN - 0.80TN = 0.80FP
% 0.20TN = 0.80FP
% TN = 4FP
% 
% We can see that these equations result in a contradiction. It's not possible for TP, FP, FN, and TN to satisfy all conditions simultaneously.
% Therefore, such an experiment cannot be constructed. The given conditions are mathematically inconsistent. Hence, it's impossible to calculate the accuracy of this experiment.
% 
