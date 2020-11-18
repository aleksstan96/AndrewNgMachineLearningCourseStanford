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

		a1 = X;
		a1 = [ones(m, 1) a1];
		z2 = a1*Theta1';
		a2 = sigmoid(z2);
		a2 = [ones(size(a2,1), 1) a2];
		z3 = a2*Theta2';
		h = sigmoid(z3);
	
		ytemp = zeros(m,num_labels);
		
		for i=1:m
			for j=1:num_labels
				ytemp(i,j)=(y(i)==j); %ytemp predstavlja matricu dimenzija 5000x10 koja se sastoji samo od 0 i 1 jer je y vektor koji sadrzi stvarne cifre
			end;
		end;

		
		for i=1:m
			for j=1:num_labels
		
			J= J + (1/m)*( -1*ytemp(i,j)*log(h(i,j)) - (1-ytemp(i,j))*(log(h(i,j)-1)));
			
			end;
		end;
		
		
		f1=0;
		f2=0;
		
		for i=1:hidden_layer_size
			for j=2:input_layer_size+1
				f1 = f1 + Theta1(i,j)*Theta1(i,j);
			end;
		end;	
		
		for i=1:num_labels
			for j=2:hidden_layer_size+1
				f2 = f2 + Theta2(i,j)*Theta2(i,j);
			end;
		end;	
		
		regFactor = (lambda/(2*m))*(f1 + f2);

		J = J + regFactor;
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

	
%BACKPROPAGATION ALGORITAM IMPLEMENTACIJA

	delta_grad_1 = zeros(size(Theta1_grad));
	delta_grad_2 = zeros(size(Theta2_grad));

	for t = 1:m
		%prvi korak: za svaki trainig primer t dodeliti a1 = x(t) i
		%pomocu forward propagation izracunati z2, a2, z3 i a3
		
		a_1 = X(t, :); %dimenzija 1x400
		a_1 = [1 a_1];  %dimenzija 1x401
		z_2 = a_1*Theta1';  %dimenzija 1x25
		a_2 = sigmoid(z_2);  
		a_2 = [1 a_2];        %dimenzija 1x26
		z_3 = a_2*Theta2'; 	  %dimenzija 1x400
		a_3 = sigmoid(z_3);   % dimenzija 1x10
		
		%drugi korak: za svaku izlaznu jedinicu l iz poslednjeg (treceg) nivoa
		%podesiti delta_3(k) = a_3(k) - y_k
		%u ovom slucaju K=10
		
		y_k = ytemp(t, :); % y_k dimenzija 1x10
		
		delta_3 = a_3 - y_k; %ovo je vektorski zapis, dimenzija 1x10
		
		%treci korak: za skriveni nivo l = 2, 
		%podesiti delta_2 = (Theta2)'*delta_3.*sigmoidGradient(z_2)
		
		delta_2 = (delta_3*Theta2) .* sigmoidGradient([1 z_2]); %delta_3*Theta2 je dimenzija 1x26
	
		%cetvrti korak: akumulirati gradijent 
	
		delta_2 = delta_2(2:end);
		delta_grad_1 = delta_grad_1 + delta_2'*a_1;
		
		%delta_3 = delta_3(2:end); NIKAKO!!!
		delta_grad_2 = delta_grad_2 + delta_3'*a_2;
		
	
	end;
	
	%peti korak: izracunati gradijent tj parcijalni izvod J (bez regularizacije)
	%tako sto dobijene delta_grad delimo sa m
	
	Theta1_grad = (1/m)*delta_grad_1;
	Theta2_grad = (1/m)*delta_grad_2;
	


%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



	regTheta1 = Theta1(:,2:end); %izbacena prva kolona
	regTheta2 = Theta2(:,2:end);

	Theta1_grad = (1/m)*(delta_grad_1 + lambda*[zeros(size(Theta1,1),1) regTheta1]);
	Theta2_grad = (1/m)*(delta_grad_2 + lambda*[zeros(size(Theta2,1),1) regTheta2]);














% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
