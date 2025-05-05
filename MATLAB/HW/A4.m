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
