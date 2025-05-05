  
load("usps_all.mat");
%Pick the correct pictures
pic1 = data(:,1,1);
pic2 = data(:,100,1);
% Run the function below to obtain the l1 distance values
disp(['L-1 distance: ', num2str(lpdist(pic1, pic2, 1))]);
% Run the function below to obtain the l2 distance values
disp(['L-2 distance: ', num2str(lpdist(pic1, pic2, 2))]);
% Run the function below to obtain the l-inf distance values
disp(['L-infinity distance: ', num2str(linfdist(pic1, pic2))]);
% Function that calculates lp distance between two vectors
function f = lpdist(x, y, p)
f = (sum(abs(x-y).^p))^(1/p);
end
% Function that calculates l-inf distance between two vectors
function f = linfdist(x, y)
f = max(abs(x-y));
end