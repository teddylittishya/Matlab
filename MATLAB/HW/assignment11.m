%% QUESTION 1:

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
num_comp = 80;

% Project the training data onto the kernel PCA space
projected_Xtrain = project(kpca, concat_training, num_comp);

% Project the testing data onto the kernel PCA space
projected_Xtest = project(kpca, concat_testing, num_comp);

% Train a k-nearest neighbors (kNN) classifier with k=20 using the
% projected training data
knn = fitcknn(projected_Xtrain, labels_concat_training, 'NumNeighbors', 20, 'Distance', 'euclidean');

% Predict the labels for the projected testing data using the kNN
% classifier
predictions = predict(knn, projected_Xtest);

% Calculate the accuracy of the predictions
accuracy = sum(predictions == labels_concat_testing) / numel(labels_concat_testing);

% Display the accuracy of the model on the test set
disp(['Accuracy for test set: ', num2str(accuracy)])
%%
% REASON FOR THE CHOICE OF RBF KERNAL The Radial Basis Function (RBF)
% kernel, also known as the Gaussian kernel, is a popular choice in machine
% learning, especially for support vector machines and kernel PCA. In the
% USPS handwritten dataset, the RBF kernel is a good choice because it can
% effectively capture the complex structures and patterns in the image
% data, leading to a high accuracy of 94.318%. The RBF kernel allows us to
% model non-linear decision boundaries, which can be crucial when dealing
% with complex datasets like images. RBF kernel often results in good
% performance, as it can handle a wide variety of data structures. The
% gamma parameter was set to 1/256^2, which is a common choice when dealing
% with image data, as it roughly corresponds to the inverse of the number
% of pixels in an image.The RBF kernel's parameter, gamma, which controls
% the flexibility of the decision boundary. A small gamma value defines a
% large similarity radius which results in more points being grouped
% together. For a large gamma value, the points need to be very close to
% each other in order to be considered similar. This allows us to control
% the complexity of the model. This choice of gamma ensures that the kernel
% is sensitive to the differences between different handwritten digits,
% allowing for accurate classification.

%% QUESTION 2: Experimental result and explanation
% 
% Yes, it is possible to significantly lower the dimensionality of the data
% using Kernel PCA with the RBF kernel. The experiment that I carried out
% by varying the number of principal components and observing the resulting
% accuracy of the kNN classifier turns out to be a good way to test this.
% 
% From my findings, it appears that even when the number of principal
% components is reduced to 20, the accuracy remains relatively high at
% 93.273%. This suggests that a significant amount of dimensionality
% reduction is possible without a substantial loss in classification
% accuracy. ie, the experimental results includes
%
% Number of Principal
% Components	    -->                         Accuracy
%
%     80	        -->                        94.364% 85
%
%     94.318% 50	-->                        94.182% 20
%
%     93.273% 5	    -->                        74.500%
%
% However, when the number of principal components is further reduced to 5,
% the accuracy drops to 74.5%. This indicates that while dimensionality
% reduction is possible, there is a limit to how much the dimensionality
% can be reduced before the loss of information starts to significantly
% impact the performance of the classifier.
% 
% That is experiment demonstrates that Kernel PCA with the RBF kernel can
% effectively reduce the dimensionality of the USPS handwritten dataset
% while maintaining a high classification accuracy, up to a certain point.
% Beyond this point, the reduction in dimensionality starts to negatively
% impact the classifier’s performance. This underscores the importance of
% choosing an appropriate number of principal components when performing
% Kernel PCA.
%% Theroritical explanation
% The way in which Kernel PCA with the RBF kernel transforms the data:
% The RBF kernel implicitly maps the
% original data points into a higher-dimensional space, where nonlinear
% relationships among data points are better captured. This
% higher-dimensional space is infinite-dimensional, but the Kernel PCA
% algorithm effectively deals with it through eigenvalue decomposition.
% Principal Component Analysis in the Transformed Space: In the transformed
% space, Kernel PCA computes principal components that maximize variance.
% However, because of the nonlinear mapping induced by the RBF kernel,
% these principal components can capture nonlinear relationships among the
% data points more effectively compared to linear PCA. 
% 
% By choosing an appropriate value for the kernel
% parameter σ , Kernel PCA can effectively concentrate variance along a
% few principal components, leading to significant dimensionality
% reduction. The RBF kernel allows for flexible representation of the data,
% enabling Kernel PCA to capture complex structures with fewer dimensions.
% 
% In the transformed space, data points that belong to
% different classes or clusters may become more separable, as the RBF
% kernel tends to pull similar points closer together and push dissimilar
% points farther apart. 
% 
% Despite the implicit mapping to a higher-dimensional space, Kernel PCA can still be
% computationally efficient due to the kernel trick, which avoids the
% explicit computation of the feature vectors in the higher-dimensional
% space. Therefore, theoretically, using Kernel PCA with the RBF kernel
% provides a powerful tool for significant dimensionality reduction by
% effectively capturing nonlinear relationships in the data and mapping it
% to a higher-dimensional space where these relationships are more
% separable.
% 
