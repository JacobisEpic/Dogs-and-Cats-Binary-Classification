%This function takes in a data matrix X, corresponding vector 
%of labels Y, and a desired label. It outputs the the number 
%of samples with desiredlabel as n_label as well as the sample
%mean vector mu_label (as a row vector) and sample covariance 
%matrix sigma_label for the data in X whose labels in Y are equal
%to the desired label.
function [n_label,mu_label,sigma_label] = labeled_mean_cov(X,Y,desiredlabel)
    FindJitIndices = find(Y == desiredlabel); % Finds the index of desiredlabel location
    XJit = X(FindJitIndices, :); % Extract the data
    mu_label = mean(XJit); % find the average
    sigma_label = cov(XJit); % find the covariance
    [n_label jit_label] = size(XJit); % Find the size


end