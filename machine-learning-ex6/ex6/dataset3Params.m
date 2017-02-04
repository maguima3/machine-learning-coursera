function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

values = [0.01 0.03 0.1 0.3 1 3 10 30];
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

minPredictionError = 10000000000000;

for i = 1:size(values, 2)
    C_i = values(i);
    for j = 1:size(values, 2)
        sigma_j = values(j);
        model= svmTrain(X, y, C_i, @(x1, x2) gaussianKernel(x1, x2, sigma_j));
        predictions = svmPredict(model, Xval);
        predictionError = mean(double(predictions ~= yval));
        if (predictionError < minPredictionError) 
            minPredictionError = predictionError;
            C = C_i;
            sigma = sigma_j;
        end
    end
end
    
fprintf('Best value of C is %f \nBest value of sigma is %f\n', C, sigma);




% =========================================================================

end
