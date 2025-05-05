% Q3
load('usps_all.mat');
% Generate random data
x1 = randn(100,1) - 1;
x2 = randn(100,1) + 1;

% Number of nearest neighbors to consider
k = 5;

% Split data into training and testing sets
x1_train = x1(1:50);
x2_train = x2(1:50);
x1_test = x1(51:100);
x2_test = x2(51:100);

% Combine training data into a single matrix
train_data = [x1_train; x2_train];

% Combine testing data into a single matrix
test_data = [x1_test; x2_test];

% Initialize error count
error_count = 0;

% Classify each test point
for i = 1:length(x1_test)
    % Compute absolute differences between test point and training points
    abs_diff = abs(train_data - test_data(i));
    
    % Sort the absolute differences
    [~, sorted_indices] = sort(abs_diff);
    
    % Find the class for the first k points
    class_counts = zeros(2, 1); % Two classes: x1 and x2
    for j = 1:k
        if sorted_indices(j) <= 50
            class_counts(1) = class_counts(1) + 1; % Belongs to class x1
        else
            class_counts(2) = class_counts(2) + 1; % Belongs to class x2
        end
    end
    
    % Determine the predicted class based on majority vote
    [~, predicted_class] = max(class_counts);
    
    % Compare predicted class to actual class
    if i <= 50 && predicted_class ~= 1
        error_count = error_count + 1; % Error on classifying x1_train
    elseif i > 50 && predicted_class ~= 2
        error_count = error_count + 1; % Error on classifying x2_train
    end
end

% Display total number of errors on train
disp(['Total number of errors on train: ', num2str(error_count)]);
%%Q3
% Assuming data is the USPS dataset loaded into a variable called 'data'

% (a) What does data(:, :, 7) represent?
class_7_images = data(:, :, 7); % Represents all 16x16 images of handwritten digits for class 7

% (b) What does data(:, 5, 7) represent?
image_5_class_7 = data(:, 5, 7); % Represents the 5th image of handwritten digits for class 7

% (c) What does data(45, 5, 7) represent?
pixel_45_5_class_7 = data(45, 5, 7); % Represents the pixel value at row 45, column 5 of the 5th image for class 7

% (d) What does data(2, :, :) represent?
row_2_all_images_all_classes = data(2, :, :); % Represents all pixel values for row 2 of all images across all classes

% (e) What does data(:, 6, :) represent?
column_6_all_images_all_classes = data(:, 6, :); % Represents all pixel values for column 6 of all images across all classes

% Displaying the results for verification
disp('(a) data(:, :, 7) represents all 16x16 images of handwritten digits for class 7');
disp('(b) data(:, 5, 7) represents the 5th image of handwritten digits for class 7');
disp('(c) data(45, 5, 7) represents the pixel value at row 45, column 5 of the 5th image for class 7');
disp('(d) data(2, :, :) represents all pixel values for row 2 of all images across all classes');
disp('(e) data(:, 6, :) represents all pixel values for column 6 of all images across all classes');


%% Q4
% Assuming data is the USPS dataset loaded into a variable called 'data'

% Extract the first image of digit 1
image_1_1 = data(:, :, 1);

% Extract the 100th image of digit 1
image_100_1 = data(:, :, 10); % Assuming there are 10 images per digit

% Compute L1 distance
L1_distance = sum(abs(image_1_1(:) - image_100_1(:)));

% Compute L2 distance
L2_distance = sqrt(sum((image_1_1(:) - image_100_1(:)).^2));

% Compute L∞ distance
L_inf_distance = max(abs(image_1_1(:) - image_100_1(:)));

% Display the distances
disp(['L1 distance: ', num2str(L1_distance)]);
disp(['L2 distance: ', num2str(L2_distance)]);
disp(['L∞ distance: ', num2str(L_inf_distance)]);

