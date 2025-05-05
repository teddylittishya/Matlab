% Load USPS Handwritten Digits dataset
load('usps_all.mat');

% Initialize a figure for viewing the averages
figure;

% Initialize an array to store the sample means for each digit
sample_means = zeros(16, 16, 10);

% Iterate over each digit from 0 to 9
for digit = 0:9
    % Extract data for the current digit
    digit_data = data(:, :, (digit * 1100) + 1 : (digit + 1) * 1100);
    
    % Compute the sample mean for the current digit
    digit_mean = mean(digit_data, 3); % Compute mean along the third dimension
    
    % Store the sample mean
    sample_means(:, :, digit + 1) = digit_mean;
    
    % Plot the mean image in the grid
    subplot(2, 5, digit + 1); % Adjust the subplot index to start from 1
    imshow(uint8(digit_mean)); % Convert to uint8 before displaying
    title(sprintf('Digit %d', digit)); % Set title for the subplot
end
