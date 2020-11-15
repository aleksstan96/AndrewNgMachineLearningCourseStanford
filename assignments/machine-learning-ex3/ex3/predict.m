function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


% Theta1 je dimenzija 25 x 401 tj 25 x (n + 1) 
% Theta2 je dimenzija 10 x 26

X = [ones(m, 1) X]; %dodajemo vektor kolone jedinica u X, pa X sada ima dimenzije m x (n+1)

temp1 = sigmoid(Theta1*X'); % dimenzija temp1 je 25 x m

a2= temp1'; % dimenzija a2 je m x 25

a2 = [ones( size(a2,1) ,1) a2]; %dodajemo vektor kolone jedinica u a2, dakle sada je a2 dimenzija  m x 26

temp2 = sigmoid(Theta2*a2'); % temp2 je dimenzija 10 x m

h = temp2'; %h je dimenzija m x 10

for i = 1:m
	[temp3, p(i)]= max(h(i, :));
end;



% =========================================================================


end
