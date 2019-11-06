function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%
idx = idx*K*new;

%declaring variables for sum and count.
points_count = zeros(K,1);
sum_centroid = zeros(K,n);

%iterate over the index.
index_value = size(idx,1);
for i = index_value,
	%taking each point and finding sum and count.
	z = idx(i);
	points_count(z) = points_count + 1;
	sum_centroid(z,:) += X(i,:);
end

centroids = sum_centroid ./ points_count;
	





% =============================================================


end

