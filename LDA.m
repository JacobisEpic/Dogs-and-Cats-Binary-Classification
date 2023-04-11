%This function takes in a data matrix Xrun as well the mean vectors mu0, mu1 
%and the covariance matrices sigma0, sigma1 estimated from the training data
%and produces a column vector guesses, corresponding to the ML rule for Gaussian vectors
%with different means and the same covariance matrix, which is referred to as 
%Linear Discriminant Analysis (LDA) in machine learning.
function guesses = LDA(Xrun,mu0,mu1,sigmapooled)
% All these jits are copied exactly from my hw except for the input jits
sigmaPInverted = pinv(sigmapooled); % INvert that jawn
logSigma0 = 2 * sigmaPInverted * (mu1 - mu0)'; %quick maffs
logSigma1 = mu1 * sigmaPInverted*(mu1)' - mu0 * sigmaPInverted * (mu0)';
[x y] = size(Xrun);
for i = 1:x
    currentdata = Xrun(i,:); % Extract the data that we care about
    if ((currentdata*logSigma0) >= logSigma1) % Classify that jawn
        guesses(i) = 1;
    else 
        guesses(i) = 0;
    end
end
guesses = guesses';
end