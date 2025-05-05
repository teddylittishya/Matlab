% Elementary data analysis for usps data
% Load the USPS data you downloaded. The data should be 256x1100x10. The
% 256 specifies how many pixels there are in each image (we will need to
% reshape the first entry in the tuple to 16x16 to view it). The second
% specifies how many instances of a given digit there are in the dataset.
% The third specifies how many digits there are (0-9).
load('usps_all.mat');
% Open a figure for viewing the data
figure;
% Here we look at the matrix of all images of the digit 3. Feel free to
% change the last argument of data to whatever digit you wish to view
x = data(:,:,3);
% Here we just get how many elements there are (1100 in this case)
xsize = size(x)
%% sample mean
% Here we find the sample mean (average over all 3's
y = sum(x, 2)/xsize(2);
% Now convert the mean back into grayscale matrix elements
y= uint8(y);
% Now we look at the "average" 3. This is all handwritten 3's "blurred"
% together
imshow(reshape(y, [16 16]));
%% sample covariance matrix
% Note: because we are dealing with multi-dimensional objects here, we need
% a covariance matrix to estimate the variability of our dataset. This is
% in a sense a generalization of the variance of a random variable.
% Again, we start by finding the sample mean (xbar in this case)
x = data(:,:,3);
xbar = sum(x, 2)/xsize(2);
xbar = double(xbar);
xsize = size(x);
Q = zeros(xsize(1));
% Now, we iterate over all the data vectors to find the covariance matrix
% A link to what the sample covariance matrix is can be found at:
% https://en.wikipedia.org/wiki/Sample_mean_and_covariance
for i = 1:xsize(2)
difvec = im2double(x(:,i)) - xbar;
Qtest = difvec*difvec.';
Q = Q+ Qtest;
end
% Q is the sample covariance matrix we have in the end
Q = Q/(xsize(2)-1);
%%
% Now we have the sample mean and covariance matrices. One thing you can
% do with these is define a normal distribution modeling a single integer
% class. This can itself be used as a classifier! Try finding a distribution
% for each integer using only the first 1000 data points, and then test
% your results on the remaining 100 data points to see for which class the
% remaining integers have the highest probability of falling into.
