function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

%defining negative and positive values.
pos =  find(y == 1);
neg = find(y == 0);


%plotting positive data in both columns
plot(X(pos,1),X(pos,2),'g+','LineWidth',2,'MarkerSize',7);

%plotting positive data in both columns
plot(X(neg,1),X(neg,2),'yo','MarkerFaceColor','r','MarkerSize',7);






% =========================================================================



hold off;

end
	