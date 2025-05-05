%% QUESTION 1.Compute  norm 2 distance from the mean of first 100 images in the corresponding class
    % find the least, average & largest distance  from the average image
% Load the dataset to the matplab workspace
load("usps_all.mat");
% Initialize arrays for storing sample mean, first 100 instances, and their distances for each digit
sample_mean_digits = zeros(256, 10);
first_100_instances = data(:, 1:100, :);
distances_to_mean = zeros(100, 10);

% Iterate over all the digits to compute sample mean for each digit
for digit = 1:10
    digit_data = data(:, :, digit);
    digit_size = size(digit_data);
    mean_vector = sum(digit_data, 2) / digit_size(2);
    mean_vector = uint8(mean_vector);
    sample_mean_digits(:, digit) = mean_vector;
end

% Calculate L2 distance between first 100 instances and sample mean of each digit
for digit = 1:10
    for instance = 1:100
        distances_to_mean(instance, digit) = norm(double(sample_mean_digits(:, digit)) - double(first_100_instances(:, instance, digit)), 2);
    end
end

% Print Least, Mean, and Largest L2 distance for each digit
for digit = 1:10
    disp(['Least L-2 distance for ', num2str(digit), ' digit: ', num2str(min(distances_to_mean(:, digit)))]);
    disp(['Mean L-2 distance for ', num2str(digit), ' digit: ', num2str(mean(distances_to_mean(:, digit)))]);
    disp(['Largest L-2 distance for ', num2str(digit), ' digit: ', num2str(max(distances_to_mean(:, digit)))]);
    fprintf('\n')
end
%% QUESTION 2: Compute  norm 2 distance for the first 100 images for each 10 classes with the overall digit mean

% Initialize array
distances_to_overall_mean = zeros(100, 10);
mean_vector_size = size(sample_mean_digits);
overall_mean = sum(sample_mean_digits, 2) / mean_vector_size(2);

% Calculate L2 distance between first 100 instances and overall sample mean of each digit
for digit = 1:10
    for instance = 1:100
        distances_to_overall_mean(instance, digit) = norm(double(overall_mean(:, 1)) - double(first_100_instances(:, instance, digit)), 2);
    end
end

% Print Least, Mean, and Largest L2 distance for each digit
for digit = 1:10
    disp(['Least L-2 distance for ', num2str(digit), ' digit: ', num2str(min(distances_to_overall_mean(:, digit)))]);
    disp(['Mean L-2 distance for ', num2str(digit), ' digit: ', num2str(mean(distances_to_overall_mean(:, digit)))]);
    disp(['Largest L-2 distance for ', num2str(digit), ' digit: ', num2str(max(distances_to_overall_mean(:, digit)))]);
    fprintf('\n')
end

%% QUESTION: 3  To find the 20 nearest neigbours for norm 2 distance b/w first 100 classes and their corresponding means for all digits


load('usps_all.mat');
num_neighbors = 20; % number of nearest neighbors
% Iterate over all the digit classes (0-9)
for digit = 1:10
    % Extract the matrix of images for the current digit
    digit_images = data(:,:,digit);
    % Calculate the size of the digit_images matrix
    num_images = size(digit_images, 2);
    % Compute the sample mean (average) for the current digit
    digit_mean = double(sum(digit_images, 2) / num_images);
    % Select the first 100 images for the current digit
    first_hundred_images = double(data(:, 1:100, digit));
    % Compute the L2 distances between the mean and the first 100 images
    l2_distances = sqrt(sum(((digit_mean - first_hundred_images).^2)));
    % Use sort() to find the indices of the 20 nearest neighbors
    [~, neighbor_indices] = sort(l2_distances, 'ascend');
    neighbor_indices = neighbor_indices(1: num_neighbors + 1); % Choose 20 + 1 neighbors
    % Remove the index of the mean itself
    if neighbor_indices(1) == 1
        neighbor_indices = neighbor_indices(2:end);
    end
    fprintf('Class %d Nearest Neighbors Distances:\n', digit);
    % Print the distances of the nearest neighbors
    for j = 1:num_neighbors
        fprintf('%.4f\n', (neighbor_indices(j)));
    end
    fprintf('\n');
end

%% QUESTION 4: Compute norm 1 distance for first 100 images and their respective mean
% Initialize array for storing L1 distances between first 100 instances and their respective means for each digit
distances_L1 = zeros(100, 10);

% Calculate L1 distance between first 100 instances and sample mean of each digit
for digit = 1:10
    for instance = 1:100
        distances_L1(instance, digit) = norm(double(sample_mean_digits(:, digit)) - double(first_100_instances(:, instance, digit)), 1);
    end
end

% Print Least, Mean, and Largest L1 distance for each digit
for digit = 1:10
    disp(['Least L-1 distance for ', num2str(digit), ' digit: ', num2str(min(distances_L1(:, digit)))]);
    disp(['Mean L-1 distance for ', num2str(digit), ' digit: ', num2str(mean(distances_L1(:, digit)))]);
    disp(['Largest L-1 distance for ', num2str(digit), ' digit: ', num2str(max(distances_L1(:, digit)))]);
    fprintf('\n');
end

%% QUESTION 5: Find 20 nearest neighbours for norm 1 distance between first 100 images & their corresponding means for each digit

% Set the value for k to retrieve the top 20 datapoints
k_value = 20;

% Initialize arrays for storing sorted distances and indices
sorted_distances_L1 = zeros(100, 10);
sorted_indices_L1 = zeros(100, 10);

% Initialize arrays for storing top k distances and indices
top_k_distances_L1 = zeros(k_value, 10);
top_k_indices_L1 = zeros(k_value, 10);

% Sort and retrieve top k distances and indices for norm-1 distance
for digit = 1:10
    [sorted_distances_L1(:, digit), sorted_indices_L1(:, digit)] = sort(distances_L1(:, digit), 'ascend');
    top_k_indices_L1(:, digit) = sorted_indices_L1(1:k_value, digit);
    top_k_distances_L1(:, digit) = sorted_distances_L1(1:k_value, digit);
end

% Compare top k indices between norm-1 and norm-2 distances
for digit = 1:10
    disp(['Digit ', num2str(digit), ': '])
    disp(top_k_indices_L1(:, digit))
    disp(top_k_indices(:, digit))
    difference_set = setdiff(top_k_indices_L1(:, digit), top_k_indices(:, digit));
end





  
