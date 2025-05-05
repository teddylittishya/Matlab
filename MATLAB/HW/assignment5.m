% SOLUTION 1
load("usps_all.mat");

% Merge the resulting 2000 entries into a single matrix of size 256 × 2000 that will be used as your train set.
usps_train_class1 = double(data(:,1:1000,3)')./255;
usps_train_class2 = double(data(:,1:1000,8)')./255;

% Merge the train set images into a single matrix
%Extract the remaining 200 images for each class
usps_test_class1 = double(data(:,1001:1100,3)')./255;
usps_test_class2 = double(data(:,1001:1100,8)')./255;

% Concatinate
usps_train = cat(1, usps_train_class1, usps_train_class2)';
usps_test = cat(1, usps_test_class1, usps_test_class2)';
%% SOLUTION 2: Draw test(:, 2) and train (:, 1002) using imshow
figure;
% Display the 1002nd image from the train set
subplot(1,2,2);
class2_digit = usps_train(:, 1002);
imshow(reshape(class2_digit, [16 16]));
% Display the second image from test data
subplot(1,2,1);
class1_digit = usps_test(:, 2);
imshow(reshape(class1_digit, [16 16]));


%% SOLUTION 3: Hard SVM
    %==>  Transpose the train and test matrices
    % ==> Convert the matrices to type double
    % ==> Train the SVM model using 'fitcsvm'
    % ==> Make predictions using the trained model.
    % ==> Count the number of mislabelled entries.

% Set the number of samples
samples_count = 1100;
class1_labels = zeros(samples_count, 1);
class2_labels = ones(samples_count, 1);

% Concatenate training and test labels
train_labels = cat(1, class1_labels(1:1000), class2_labels(1:1000));
test_labels = cat(1, class1_labels(1001:end), class2_labels(1001:end));

% Fit a Hard SVM model to the data
svm_model = fitcsvm(usps_train', train_labels, 'Standardize',true,'KernelFunction','linear','KernelScale','auto','BoxConstraint', Inf);
svm_model = fitPosterior(svm_model, usps_train',train_labels);
[labels, posterior] = predict(svm_model, usps_test');
mislabeled = sum(labels ~= test_labels);
disp(['Mislabeled Points: ', num2str(mislabeled)]);
% Calculate accuracy
accuracy = sum(labels == test_labels) / length(test_labels) * 100;
disp(['Accuracy: ', num2str(accuracy), '%']);

% I got an increase in accuracy from 93.5 % to 95.5 when I tried
% experimenting with additional parameters such as standardization, kernel function selection, and automatic scaling



