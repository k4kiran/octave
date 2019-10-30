function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

% Adding column vector of ones to the X.
X = [ones(m,1) X];

% Defining the values of each variables.
a1 = X;

% Layer2 variables
z2 = Theta1 * a1';
a2 = sigmoid(z2);
%disp(size(a2));
%disp(a2);

% Layer3 Variables
a2 = [ones(m,1) a2'];
%disp(size(a2));
z3 =  Theta2 * a2';

%hypothesis
h = sigmoid(z3);
%disp("final hypothesis is");
%disp(h);

% Initializing final output vector with values as zero 
y_final = zeros(num_labels,m);

% Changing the values to 1 where neccesary
for i = 1:m,
	%disp("new values are");
	%disp(y(i));
	row = y(i);
	y_final(row,i) = 1;
end

%calculating the cost function
%term1 = (-y_final) .* log(h);
%disp("here");
%disp(size(h));
%disp(size(y_final'));

%finding individual terms in the cost equation
term1 = -y_final .* log(h);
term2 = (1 - y_final) .* log(1 - h);
final_term = term1 - term2;

%cost function
J = (1/m) * (sum(sum(final_term)));
%disp("cost is");
%disp(J);


%Regualrization term calculation

%exclude first row(bias) from theta1(i,j) and theta2(i,j)
theta_ij_layer1 = Theta1(:,2:end);
theta_ij_layer2 = Theta2(:,2:end);

% sum calcualtion in layer1.
sum_term1 = sum(sum(theta_ij_layer1 .^ 2));
sum_term2 = sum(sum(theta_ij_layer2 .^ 2));


%calculating regularization term
reg_term = (lambda/(2 * m)) * (sum_term1 + sum_term2); 

%regularizing cost function
J = J + reg_term;

	
	

%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

for k = 1:m,
	
	% step1
	%calculate terms in layer2 
	a1  = X(k,:)';	
	z2 = Theta1 * a1;
	a2 = sigmoid(z2);
	
	%adding bias 
	a2 = [1;a2];
	
	%calculate terms in layer3
	z3 = Theta2 * a2;
	a3 = sigmoid(z3);
	
	%step2
	%calculate error in the last layer
	delta3 = a3 - y_final(:,k);
	%disp("here");
	%disp(delta3);
	z2 = [1;z2];
	
	%step3
	%calculate error in second last layer
	delta2 = (Theta2') * delta3 .* sigmoidGradient(z2);
	delta2 = delta2(2:end);
	
	%step4
	%calculating D(i,j) in reverse
	Theta2_grad = Theta2_grad + delta3 * a2';
	Theta1_grad = Theta1_grad + delta2 * a1';
end;
	
	%step5
	%derivative of gradient
	disp("Training in Progress ....... Please wait....");
	Theta2_grad = (1/m) * Theta2_grad;
	Theta1_grad = (1/m) * Theta1_grad;
	
	
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%adding regularization to 2 to end rows
Theta1_grad(:,2:end) =  Theta1_grad(:,2:end) + ((lambda/m) * Theta1(:,2:end));
Theta2_grad(:,2:end) =  Theta2_grad(:,2:end) + ((lambda/m) * Theta2(:,2:end));

%unrolling the vectors
grad = [Theta1_grad(:);Theta2_grad(:)];


% -------------------------------------------------------------

% =========================================================================


end
