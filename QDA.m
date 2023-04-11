%This function takes in a data matrix Xrun as well the mean vectors mu0, mu1 
%and the covariance matrices sigma0, sigma1 estimated from the training data
%and produces a column vector guesses, corresponding to the ML rule for Gaussian vectors
%with different means and different covariance matrices, which is referred to as 
%Quadratic Discriminant Analysis (QDA) in machine learning.
function guesses = QDA(Xrun,mu0,mu1,sigma0,sigma1)
    s0Inverted = pinv(sigma0); % Invert that jawn
    s1Inverted = pinv(sigma1);
    s0LE = sum(log(eig(sigma0))); % Find the sum
    s1LE = sum(log(eig(sigma1)));
    [x y] = size(Xrun); % This size jawn be trippin but we just need
    % x = length(Xrun);    % I don't know why this doesn't work
    for i = 1:x
        currentdata = Xrun(i,:); %Extract data we want
        LL0 = -0.5 * ((currentdata - mu0) * s0Inverted * (currentdata - mu0)' + s0LE);
        LL1 = -0.5 * ((currentdata - mu1) * s1Inverted * (currentdata - mu1)' + s1LE);
        if (LL0 <= LL1) % Classify that jawn
            guesses(i) = 1;
        else if (LL0 > LL1)
            guesses(i) = 0;
        end
    end
        guesses = guesses'; % Transpose this jawn

end