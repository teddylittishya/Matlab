%% sample_code_usps_data.m
load("usps_all.mat");
%% dataset is 256x1100x10 ie, pixels(reshape to 16 x16) x instances of digits x digits(0-9)
figure;
imshow(reshape(data(:,5,3),[16 16]),[]);

%% sample_code_usps_EDA.m
x = data(:,:,3);
xsize = size(x)

%==> Sample mean
y = sum(x, 2)/xsize(2);
% Convert the mean aback into greyscale matrix images
y = uint8(y);
imshow(reshape(y, [16 16])); % avg of all 3's and its blurred

% ==> Sample covariance matrix -estimate the vatriability of dataset
x = data(:,:,3);
xbar = sum(x, 2)/xsize(2);
xbar = double(xbar);
xsize = size(xbar);
Q = zeros(xsize(1));
% Now we iterate over all the data vectors
for i = 1:size(2)
    difvec = im2double(x(:,i)) - xbar;
    Qtest = difvec*difvec.';
    Q = Q +Qtest;
end
% Q is the sample coivariance matrix
Q = Q /(xsize(2)-1);

%% sample_code_svm1.m
%vectors in class 0,1
clear all
close all
n = 1000;
theta0 = 2*pi*rand(n,1);
theta1 = 2*pi*rand(n,1);
x0 = rand(n,1).*[cos(theta0),sin(theta0)];
x1 = (rand(n,1) + 1).*[cos(theta1),sin(theta1)];
y0 = zeros(n,1);
y1 = ones(n,1);
X = vertcat(x0,x1);
Y = vertcat(y0,y1);
% Fit the Model
Mdl = fitcsvm(X,Y, 'KernelFunction','RBF','KernelScale','auto');
sv = Mdl.SupportVectors;
figure
hold on
% Plot the original data
gscatter(X(:,1),X(:,2),Y)
%plot the support vectors from our model
plot(sv(:,1),sv(:,2),'ko','MarkerSize',10)
legend('class 0','class 1', 'Support Vector')
hold off

%% sample_code_svm2.m

%vectors in class 0,1
n = 1000;
x0 = rand(n,1) -1;
x1 = rand(n,1);
x0 = horzcat(x0, rand(n,1));
x1 = horzcat(x1, rand(n,1));
y0 = zeros(n,1);
y1 = ones(n,1);
X = vertcat(x0,x1);
Y = vertcat(y0,y1);
Mdl = fitcsvm(X,Y)

sv = Mdl.SupportVectors;
figure
hold on
for i = 1:n
    plot(x0(i,1), x0(i,2), 'bo')
    plot(x1(i,1), x1(i,2), 'ro')
end
plot(sv(:,1),sv(:,2),'ko', 'MarkerSize',10)
legend('class 0','class 1','Support Vector')
hold off

%% sample_code_svmUSPS.m
clear all;
load('usps_all.mat');
x0 = double(data(:,1:1000,10)')./255;
x1 = double(data(:,1:1000,1)')./255;
x2 = double(data(:,1:1000,2)')./255;
x3 = double(data(:,1:1000,3)')./255;
x4 = double(data(:,1:1000,4)')./255;
x5 = double(data(:,1:1000,5)')./255;
x6 = double(data(:,1:1000,6)')./255;
x7 = double(data(:,1:1000,7)')./255;
x8 = double(data(:,1:1000,8)')./255;
x9 = double(data(:,1:1000,9)')./255;

% Conncatinate all the data- necessary fro fitcsvm
X = vertcat(x0,x1,x2,x3,x4,x5,x6,x7,x8,x9);
xsize = size(x0);
y0 = zeros(xsize(1),1);
y1 = ones(xsize(1),1);
y2 = ones(xsize(1),1);
y3 = ones(xsize(1),1);
y4 = ones(xsize(1),1);
y5 = ones(xsize(1),1);
y6 = ones(xsize(1),1);
y7 = ones(xsize(1),1);
y8 = ones(xsize(1),1);
y9 = ones(xsize(1),1);

% Concatinate labels
Y = vertcat(y0,y1,y2,y3,y4,y5,y6,y7,y8,y9);
% Fit a model to the data
Mdl0 = fitcsvm(X,Y,'Standardize',true,'KernelFunction','RBF','KernelScale','auto');
Mdl0 = fitPosterior(Mdl0,X,Y);
testpt = double(data(:,1001,10)')/255;
[labels, posts] = predict(Mdl0,testpt)

%% Bivariate_Normal.m
mu = [2,3];
Sigma = [1 .5; .5 1];
sim = mvnrnd(mu,Sigma,10000);
figure;
plot(sim(:, 1), sim(:, 2), '.');

%% Classifier_Normality.m
load('usps_all.mat');
%train/test data split
train = data(:,1:1000,3);
test = data(:,1001:1100, 3);

train_size = size(train);
test_size = size(test);

%sample mean
avg_three = sum(train, 2)/train_size(2);
avg_three = uint8(avg_three);
figure;
imshow(reshape(avg_three , [16 16]));

%Sample covariance matrix
xbar = double(avg_three);
Q = zeros(train_size(1));
for i = 1:train_size(2)
    difvec = im2double(train(:,i)) - xbar;
        Qtest = difvec*difvec.';
        Q = Q + Qtest;
end
Q = Q/(train_size(2)-1);
disp(xbar(1));
disp(Q(1,1));
figure;
imshow(reshape(uint8(Q), [256 256]));

% Generative Models
y = mvnrnd(xbar, Q, 5);

for i = 1:5
    z = y(i, :);
    z = uint8(z);
    figure;
    imshow(reshape(z, [16 16]));
end

% Clasifier
train4 = data(:, 1:1000,4);
test4 = data(:, 1001:1100, 4);
train_size4 = size(train4);
test_size4 = size(test4);
avg_four = sum(train4, 2)/train_size4(2);
avg_four = uint8(avg_four);
xbar4 = double(avg_four);

% We now classify 3s and 4s.
for i = 1:100
    z1 = double(test4(:, i));
    thr1 = threshold(z1, xbar, xbar4, Q);
    if thr1 > 0
        disp(4);
    else
        disp(3);
    end
end
for i = 1:100
    z1 = double(test(:, i));
    thr1 = threshold(z1, xbar, xbar4, Q);
    if thr1 > 0
        disp(4);
    else
        disp(3);
    end
end

% Function to compute thresholds
function f = threshold(n, avg3, avg4, cov)
    f = (.5)*(transpose(n-avg3)*inv(cov)*(n-avg3) - transpose(n-avg4)*inv(cov)*(n-avg4));
end

%% Goal: Find the neighborhood sets of a point under different metrics
%%%
% Data set consists of two groups of points, each in the shape of a
% semi-ellipse.
rng('default') % for reproducible random generation
n = 500; % number of points on each group
t = linspace(0, 1, n);
t = t';
% random noise added to semi-ellipse
lx_noise = 2*rand(n, 1);
ly_noise = 2*rand(n, 1);
rx_noise = 2*rand(n, 1);
ry_noise = 2*rand(n, 1);
% centers of semi-ellipses, written as row vectors
leftCenter = [-2*ones(n, 1), -1*ones(n, 1)];
rightCenter = [2*ones(n, 1), 1*ones(n, 1)];
% semi-ellipses on the left and right, each point is a row vector
leftMoon = [4*cos(pi*t) + lx_noise, 6*sin(pi*t) + ly_noise] + leftCenter;
rightMoon = [4*cos(pi + pi*t) + rx_noise, 6*sin(pi + pi*t) + ry_noise] +
rightCenter;
figure,
scatter(leftMoon(:, 1), leftMoon(:, 2), [], 'b', 'filled', 'd')
hold on
scatter(rightMoon(:, 1), rightMoon(:, 2), [], 'r', 'filled', 'd')
axis equal
hold off
title('Two Moons')
%
%%%
% We can examine the effect that the choice of distance measure has on the
% set of neighbors of a point.
% Combine both groups into one array:
X = [leftMoon; rightMoon];
k = 50; % number of neighbors
% Choose a point on the edge of the left moon:
xi = 5;
x = X(xi, :);
%%%
% Starting with the Euclidean $L^2$ distance, we can find the $50$
% nearest neighbors to the point $x$.
%%%
% In subtracting the vector $x$ with the matrix $X$, MATLAB automatically
% subtracts $x$ from each row of $X$.
% Subtract, square each entry, get the square norm of each difference, sqrt
xDist2 = sqrt(sum((x - X).^2, 2));
% Use sort() to get the indices of the 50 distances
[~, xNeighbors2] = sort(xDist2, 'ascend');
xNeighbors2 = xNeighbors2(1: k + 1); % Choosing 50 + 1 neighbors
isequal(xNeighbors2(1), xi)
%
% the smallest distance is between x and itself, so we get rid of that
xNeighbors2 = xNeighbors2(2:end);
%
% Make colors to show neighborhood membership
% using RGB colors
moonColors = [0*ones(n, 1), 0*ones(n, 1), ones(n, 1);...
ones(n, 1), 0*ones(n, 1), 0*ones(n, 1)];
% The neigborhood of x is yellow:
moonColors(xNeighbors2, :) = [zeros(k, 1), 0.5*ones(k, 1), 0.5*ones(k, 1)];
moonColors(xi, :) = [0 0 0]; % x itself is black
xRadius2 = norm(x - X(xNeighbors2(k), :), 2); % distance between x and its furthest
neighbor
cx = x(1) + xRadius2*cos(2*pi*t);
cy = x(2) + xRadius2*sin(2*pi*t);
figure,
scatter(leftMoon(:, 1), leftMoon(:, 2), [], moonColors(1:n, :), 'filled', 'd')
hold on
scatter(rightMoon(:, 1), rightMoon(:, 2), [], moonColors(n+1:end, :), 'filled',
'd')
plot(cx, cy, 'Color', 'k')
axis equal
hold off
title('$$ L^2 $$ Neighborhood', 'interpreter', 'latex')
snapnow
%
%%%
% We can repeat this process using the $L^1$ distance:
xDist1 = sum(abs(x - X), 2);
[~, xNeighbors1] = sort(xDist1, 'ascend');
xNeighbors1 = xNeighbors1(1: k + 1); % Choosing 50 + 1 neighbors
isequal(xNeighbors1(1), xi)
xNeighbors1 = xNeighbors1(2:end);
%
moonColors = [0*ones(n, 1), 0*ones(n, 1), ones(n, 1);...
ones(n, 1), 0*ones(n, 1), 0*ones(n, 1)];
% The neigborhood of x is teal:
moonColors(xNeighbors1, :) = [zeros(k, 1), 0.5*ones(k, 1), 0.5*ones(k, 1)];
moonColors(xi, :) = [0 0 0]; % x itself is black
xRadius1 = norm(x - X(xNeighbors1(k), :), 1); % distance between x and its furthest
neighbor
%%%
% In Euclidean distance, a circle is formed by taking all the points
% equidistant from a fixed point. But using the $L^1$ distance, that isn't
% the case. Instead, the equidistant points form a diamond-like shape.
% Getting the vertices of the diamond, moving clockwise from left
dx = x(1) + xRadius1*[-1, 0, 1, 0, -1];
dy = x(2) + xRadius1*[0, 1, 0, -1, 0];
figure,
scatter(leftMoon(:, 1), leftMoon(:, 2), [], moonColors(1:n, :), 'filled', 'd')
hold on
scatter(rightMoon(:, 1), rightMoon(:, 2), [], moonColors(n+1:end, :), 'filled',
'd')
line(dx, dy, 'LineStyle', '-', 'Color', 'k', 'LineWidth', 1)
axis equal
hold off
title('$$ L^1 $$ Neighborhood', 'interpreter', 'latex')
snapnow
%
%%%
% Again, we can repeat this for another $p$-norm distance. This time, we
% will find the $$ L^{\infty} $$ neighborhood of $x$.
xDistInf = max(abs(x - X), [], 2);
[~, xNeighborsInf] = sort(xDistInf, 'ascend');
xNeighborsInf = xNeighborsInf(1: k + 1); % Choosing 50 + 1 neighbors
isequal(xNeighborsInf(1), xi)
%
xNeighborsInf = xNeighborsInf(2:end);
moonColors = [0*ones(n, 1), 0*ones(n, 1), ones(n, 1);...
ones(n, 1), 0*ones(n, 1), 0*ones(n, 1)];
% The neigborhood of x is teal:
moonColors(xNeighborsInf, :) = [zeros(k, 1), 0.5*ones(k, 1), 0.5*ones(k, 1)];
moonColors(xi, :) = [0 0 0]; % x itself is black
xRadiusInf = norm(x - X(xNeighborsInf(k), :), Inf); % distance between x and its
furthest neighbor
%%%
% The $$ L^{\infty} $$ distance has equidistant sets that look like squares.
% getting vertices of the square, moving clockwise from top left
sx = x(1) + (xRadiusInf)*[-1, 1, 1, -1, -1];
sy = x(2) + (xRadiusInf)*[1, 1, -1, -1, 1];
figure,
scatter(leftMoon(:, 1), leftMoon(:, 2), [], moonColors(1:n, :), 'filled', 'd')
hold on
scatter(rightMoon(:, 1), rightMoon(:, 2), [], moonColors(n+1:end, :), 'filled',
'd')
line(sx, sy, 'LineStyle', '-', 'Color', 'k', 'LineWidth', 1)
axis equal
hold off
title('$$ L^{\infty} $$ Neighborhood', 'interpreter', 'latex')
snapnow
