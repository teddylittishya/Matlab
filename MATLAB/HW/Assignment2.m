
% To load USPS Handwritten Digits dataset
% This line loads the USPS Handwritten Digits dataset from the file "USPS_all.mat" into MATLAB's workspace.
load("USPS_all.mat");
%% QUESTION 1: PLOT 2 X 5 IMAGE GRID OF IMAGES
% 1. Download the USPS Handwritten Digits data set (usps all) from the website of Prof. Roweis. Please produce a 2 by 5 image grid of images: the 1st digit one, the 2nd digit two, the 3rd digit 3, ..., the 9th nine and the 10th zero from the set:
% (1 2 3 4 5!
%  6 7 8 9 0)

% Inorder to view the data,first open a figure
% This command initializes a new figure window to display the image
figure;
% For loop to iterate over all the digits to plot in 2x5 grid
% This starts a loop that iterates from 1 to 10, 
% corresponding to the 10 digits (0 to 9).
% Inside the loop, 'subplot(2,5,i)' divides the figure
% window into a 2 by 5 grid of subplots, and
% 'i' determines which subplot the current digit 
% will be plotted in.
for i = 1:10
% create a grid of subplots with 2 rows and 5 columns, 
% and the i variable iterates from 1 to 10. 
% This means there will be a total of 10 subplots 
% arranged in a grid of 2 rows and 5 columns.
    subplot(2,5,i);   
    x = data(:,i,i); 
   
    % This line extracts image data for the current
    % digit (i) from the dataset.
    % Assuming data is a 3-dimensional array,
    % the expression data(:,i,i) selects the image 
    % data for the ith digit. 
    % The colon : represents all rows, 
    % i represents the ith column, and
    % i again represents the ith slice along the 
    % third dimension.
    imshow(reshape(x,[16 16])); 
    % This command reshapes the image data into a 
    % 16 by 16 grid and displays it using the 
    % imshow function. 
    % This function displays the grayscale image
    % represented by the matrix x.

% The loop continues until all 10 digits have 
% been plotted in the 2 by 5 grid.
% This script produces a 2 by 5 image grid, 
% where each row represents a digit from 0 to 9,
% with the images displayed in order. 
% Each image corresponds to a handwritten digit 
% from the USPS dataset.
end

%% QUESTION 2: PLOT THE SAMPLE MEAN FOR ALL DIGITS IN THE 2X5 GRID
% open figure to view data
figure;
% Array to store the samples means for digits from 1 to 9 and 0
sample_means = zeros(256, 10);
% 
for i = 1:10
    subplot(2,5,i);
    data_digit = data(:,:,i);
    xsize = size(data_digit);
    data_mean = sum(data_digit,2)/xsize(2);
    data_mean = uint8(data_mean);
    sample_means(:, i) = data_mean;
    imshow(reshape(data_mean,[16 16]));
end


%% QUESTION 3: PLOT THE SAMPLE MEAN FOR ALL 10 FULL(1100 SAMPLES) SETS
% --> AVERAGE OF THE HANDWRITTEN DIGITS
% --> RESHAPE AS 16X16 & INCLUDE THIS OVERALL AVG MEAN
% Initialize figure
figure;
sample_means_size = size(sample_means);
% -> Compute the sample mean for the current digit
total_sample_mean = sum(sample_means,2)/sample_means_size(2);
total_sample_mean = uint8(total_sample_mean);
imshow(reshape(total_sample_mean,[16 16]));

%% QUESTION 4: SAMPLE COVARIENCE MATRICES FOR EACH OF THE 10 SETS OF 1100 DIGIT EXAMPLES
figure;
% Here we define a 3D sample covarience matrix
d_array = zeros([256 256 10]);
% For each digit we calculate the sample covariance 
for i=1:sample_means_size(2)
    for j=1:xsize(2)
        data_digit = data(:,:,i);
        vector_difference = im2double(data_digit(:,j)) - sample_means(:, i);
        outer_product = vector_difference*vector_difference.';
        d_array(:,:,i) = d_array(:,:,i) + outer_product;
    end
    d_array(:, :, i) = d_array(:, :, i)/(xsize(2) - 1);
end

% Plot each digit sample covarience in 2x5 grid
for i=1:sample_means_size(2)
    subplot(2,5,i);
    imshow(reshape(uint8(d_array(:,:,i)),[256 256]));
end



%% QUESTION 5: PLOT OF THE 120TH CENTRAL PIXEL's MARGINAL DISTRIBUTION FOR EACH 10 DIT CLASSES USING 2X5 IMAGE GRID
% HERE WE GENERATE A MULTIVARIATE NORMAL DISTRIBUTION

% Initialize a new figure
% Define the number of samples
figure;
no_samples = 10000;

% Initialize a 3D array to store the samples for each digit class
% Initialize an array to store the samples of the 120th central pixel for each class
arr_samples = zeros([no_samples, 256, sample_means_size(2)]);
distribution = zeros([no_samples, sample_means_size(2)]);

% Generate samples for each digit class
for i = 1:sample_means_size(2)
    % Generate samples from a multivariate normal distribution using mvnrnd
    arr_samples(:, :, i) = mvnrnd(sample_means(:, i), d_array(:, :, i), no_samples);
    
    % Extract the values of the 120th central pixel from each sample
    distribution(:, i) = arr_samples(:, 120, i);
end

% Convert arrays to uint8 for histogram plotting
% Define histogram edges
arr_samples = uint8(arr_samples);
distribution = uint8(distribution);
hist_edges = 0:5:255;

% Plot histograms for each digit class
for i = 1:10
    % Create subplot for each digit class
    % Plot histogram of marginal distribution of the 120th central pixel
    % Set title, axis labels, and limits
    subplot(2, 5, i);
    histogram(distribution(:, i), hist_edges, 'Normalization', 'probability');
    title(['Digit ', num2str(mod(i, 10))]);
    xlabel('Intensity');
    ylabel('Probability');
    xlim([0 255]);
end
