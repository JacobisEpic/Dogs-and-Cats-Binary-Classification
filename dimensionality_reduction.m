%This function takes in a data matrix Xrun, mean vector mu, 
%eigenvector matrix V, and eigenvalues D, and dimension k. 
%It selects the k eigenvectors corresponding to the k largest
%eigenvalues, centers the data by subtracting mu, and projects
%the centered data to k dimensions by multiplying by the matrix
%of k eigenvectors.
function Xrun_reduced = dimensionality_reduction(Xrun,mu,V,D,k)
    FoundedJits = flip(find(sort(-diag(D)))); % Finds the indices of the eigenvalues in the D matrix
    VectorJit = V(:,FoundedJits(1:k)); % Stores the top 10 eigenvalues
    Xrun_reduced = (Xrun - mu) * VectorJit; % Outputs the result
end