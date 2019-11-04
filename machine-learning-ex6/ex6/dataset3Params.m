function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

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


sample_values = [0.01,0.03,0.1,0.3,1,3,10,30]';
min_error = 1;
for i = 1:8,
	for j = 1:8,
		c_sample = sample_values(i);
		s_sample = sample_values(j);
		
		%x1 = [1];
		%x2 = [2];
		
		%training the model using the gaussian kernal
		mymodel = svmTrain(X,y,c_sample,@(x1,x2) gaussianKernel(x1, x2, s_sample));
		
		%finding the predictions from the model
		predictions = svmPredict(mymodel, Xval);
		
		%finding the prediction error
		pred_error = mean(double(predictions ~= yval));
		
		%getting optimal values for the parameters.
		if pred_error < 1,
			C = c_sample;
			sigma = s_sample;
			%min_error = pred_error;
			
		end
		
		
	end
end




% =========================================================================

end
