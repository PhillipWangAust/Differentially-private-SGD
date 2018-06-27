function [hiddenWeights, outputWeights, objValues] = trainStochasticSquaredErrorTwoLayerPerceptron(activationFunction, dActivationFunction, u, inputValues, targetValues, maxIter, batchSize, learningRate, privacy, C, epsilon)
% trainStochasticSquaredErrorTwoLayerPerceptron Creates a two-layer perceptron
% and trains it on the MNIST dataset. Stops after 1 epoch.
%
% INPUT:
% activationFunction             : Activation function used in both layers
% dActivationFunction            : Derivative of the activation
% function used in both layers
% u                              : Number of hidden units
% inputValues                    : Input values for training
% targetValues                   : Target values for training
% maxIter                        : Number of iterations to train
% batchSize                      : Plot error after batchSize images
% learningRate                   : Learning rate multiplier
% privacy                        : 1 for private algorithm, 0 otherwise
% C                              : Desired global sensitivity of each step
% epsilon                        : Privacy budget
%
% OUTPUT:
% hiddenWeights                  : Weights of the hidden layer
% outputWeights                  : Weights of the output layer
% objValues                      : Error values of the selected iterations
% 
    %% setup
    s = size(inputValues, 2);
    d = size(inputValues, 1);
    o = size(targetValues, 1);
    
    % Initialize the weights for the hidden layer and the output layer.
    hiddenWeights = rand(u, d);
    outputWeights = rand(o, u);
    
    hiddenWeights = hiddenWeights./size(hiddenWeights, 2);
    outputWeights = outputWeights./size(outputWeights, 2);
        
    % plotting setup
    iterSelection = (maxIter/20):(maxIter/20):maxIter; %select 20 iterations to show error for
    iterSelection = ceil([iterSelection(1)/5 iterSelection]); %add a 21st early iteration
    objValues = zeros(1,21);
    
    % generate data permutation
    samples = randperm(s);
    
    %% PSGD algorithm
    j = 0;
    for t = 1: maxIter

        rate = learningRate/sqrt(t);

        for k = 1: batchSize
            sample = (t-1)*batchSize+k; %next sample in line
            n = samples(sample); 

            % Propagate the input vector through the network.
            inputVector = inputValues(:, n);
            targetVector = targetValues(:, n);
            hiddenActualInput = hiddenWeights*inputVector;
            hiddenOutputVector = activationFunction(hiddenActualInput);
            outputActualInput = outputWeights*hiddenOutputVector;
            outputVector = activationFunction(outputActualInput);


            % Backpropagate the errors.
            outputDelta = dActivationFunction(outputActualInput).*(outputVector - targetVector);
            hiddenDelta = dActivationFunction(hiddenActualInput).*(outputWeights'*outputDelta);

            outputUpdate = outputDelta*hiddenOutputVector';
            hiddenUpdate = hiddenDelta*inputVector';

            
            % clipping
            allUpdates = [hiddenUpdate outputUpdate'];
            nrm = norm(allUpdates(:));
            outputUpdate = outputUpdate/max(1,nrm/C);
            hiddenUpdate = hiddenUpdate/max(1,nrm/C);

            
            % generate noise and reshape vector to appropriate matrices
            Z = generateNoise(o*u+u*d,1,C/epsilon);
            Z1 = Z(1:o*u);
            Z1 = reshape(Z1,o,u); 
            Z2 = Z((o*u)+1:o*u+u*d);
            Z2 = reshape(Z2,u,d);
            outputUpdate = outputUpdate + privacy*Z1;
            hiddenUpdate = hiddenUpdate + privacy*Z2;
            
            
            % perform steps
            outputWeights = outputWeights - rate.*outputUpdate;
            hiddenWeights = hiddenWeights - rate.*hiddenUpdate;

        end;
        
        %% save error value of selected iterations
        if ismember(t,iterSelection)
            j=j+1;
            error = sum(sum((activationFunction(outputWeights*activationFunction(hiddenWeights*inputValues)) - targetValues).^2));
            error = error/s;
            objValues(1,j) = error;
        end
    end
end