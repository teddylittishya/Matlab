% Load the USPS handwritten dataset
load('usps_all.mat')

% Initialize cell arrays to store training and testing data for each digit
all_training = cell(1, 10);
all_testing = cell(1, 10);

% Initialize arrays to store training and testing labels
labels_concat_training = [];
labels_concat_testing = [];

% Loop through each digit from 1 to 10
for i = 1:10
    % Select 880 samples for training
    training = data(:, 1:880, i);
    % Select 220 samples for testing
    testing = data(:, 881:1100, i);

    % Store training and testing data in respective cell arrays
    all_training{i} = training;
    all_testing{i} = testing;

    % Create training labels by repeating the digit 'i' 880 times
    train_labels = repmat(i, 1, 880);
    labels_concat_training = [labels_concat_training, train_labels];

    % Create testing labels by repeating the digit 'i' 220 times
    testing_labels = repmat(i, 1, 220);
    labels_concat_testing = [labels_concat_testing, testing_labels];
end

% Concatenate the training and testing data across digits
concat_training = cat(2, all_training{:});
concat_testing = cat(2, all_testing{:});

% Transpose the label arrays to match the data dimensions
labels_concat_training = transpose(labels_concat_training);
labels_concat_testing = transpose(labels_concat_testing);

% Convert the data to double precision and transpose it
concat_training = transpose(double(concat_training));
concat_testing = transpose(double(concat_testing));

% Perform Kernel PCA on the training data with a Gaussian kernel
kpca = KernelPca(concat_training, 'gaussian', 'gamma', 1/256^2, 'Autoscale', true);

% Define the number of components to keep
num_comp = 70;

% Project the training data onto the kernel PCA space
projected_Xtrain = project(kpca, concat_training, num_comp);

% Project the testing data onto the kernel PCA space
projected_Xtest = project(kpca, concat_testing, num_comp);

% Train a k-nearest neighbors (kNN) classifier with k=20 using the projected training data
knn = fitcknn(projected_Xtrain, labels_concat_training, 'NumNeighbors', 20, 'Distance', 'euclidean');

% Predict the labels for the projected testing data using the kNN classifier
predictions = predict(knn, projected_Xtest);

% Calculate the accuracy of the predictions
accuracy = sum(predictions == labels_concat_testing) / numel(labels_concat_testing);

% Display the accuracy of the model on the test set
disp(['Accuracy for test set: ', num2str(accuracy)])
