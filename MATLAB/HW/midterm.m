% Open the file
train_images = fopen('/Users/teddythomas/Downloads/train-images.idx3-ubyte');
train_labels = fopen('/Users/teddythomas/Downloads/train-labels.idx1-ubyte');
test_images = fopen('/Users/teddythomas/Downloads/t10k-images.idx3-ubyte');
test_labels = fopen('/Users/teddythomas/Downloads/t10k-labels.idx1-ubyte');

%%
% Consider the N1, N2,N3 considered based on both runtime and performance metrices
% N1: A smaller training set size (e.g., 1000 images per class).
% N2: A moderate training set size (e.g., 4000 images per class).
% N3: A larger training set size (e.g., 5000 images per class).
% 
% ==>Effect on runtime
% 
% 1. Increase the training set size:
% The runtime of the KNN algorithm also increases.
% Each query instance needs to compute distances to all training instances, which becomes more time-consuming with a larger dataset.
% Larger training sets require more memory and computational resources.
% 2. Decreasing Training Set Size:
% Smaller training sets lead to faster prediction times.
% However, if the training set is too small, the model may not capture the underlying patterns well, resulting in poor generalization.
% 
% ==>Effect on Error Metrics:
% 
% Increasing Training Set Size:
% Generally, larger training sets improve the model’s ability to generalize.
% With more diverse examples, the KNN algorithm can better estimate class boundaries.
% This often leads to lower training error (better fit to the training data).
% However, overfitting can occur if the training set becomes too large, causing poor performance on unseen data.
% Decreasing Training Set Size:
% Smaller training sets may lead to higher training error due to limited representation of the data.
% The model might underfit, failing to capture complex relationships.
% Validation error may also increase due to poor generalization.
% 
% ==> Generalization Error:
% 
% Bias-Variance Trade-off:
% KNN has a bias-variance trade-off.
% Small training sets (low N) tend to have high variance (sensitive to noise).
% Large training sets (high N) tend to have low variance but may introduce bias (overfitting).
% Cross-Validation:
% Use techniques like k-fold cross-validation to estimate generalization error.
% Experiment with different training set sizes and evaluate performance on validation data.

%%
% Load Fashion MNIST dataset
[fashion_mnist_images, fashion_mnist_labels] = fashion_mnist_load_data();

% Split into training and testing sets
train_images = fashion_mnist_images.train_images;
train_labels = fashion_mnist_labels.train_labels;

test_images = fashion_mnist_images.test_images;
test_labels = fashion_mnist_labels.test_labels;

function [fashion_mnist_images, fashion_mnist_labels] = fashion_mnist_load_data()
    % Load Fashion MNIST dataset
    fashion_mnist = load('fashion_mnist.mat');
    
    % Extract images and labels
    fashion_mnist_images.train_images = fashion_mnist.train_images;
    fashion_mnist_labels.train_labels = fashion_mnist.train_labels;
    
    fashion_mnist_images.test_images = fashion_mnist.test_images;
    fashion_mnist_labels.test_labels = fashion_mnist.test_labels;
end
