function [] = applyStochasticSquaredErrorTwoLayerPerceptronMNIST()
%applyStochasticSquaredErrorTwoLayerPerceptronMNIST Train the two-layer
%perceptron using the MNIST dataset and evaluate its performance.

    
    % Choose appropriate parameters.
    d = 15; %input layer size
    u = 20; %hidden layer size
    batchSize = 100;
    learningRate = 1; %learning rate multiplier
    C = 1; %desired global sensitivity limit
    epsilon = 1; %privacy budget
    sims = 20; %amount of simulations
    
    
    % Load MNIST.
    [inputValues,~] = loadMNIST(d);
    % This algorithm is for solving the complete classification problem so
    % we don't use the 0 vs the rest target values
    labels = loadMNISTLabels('train-labels.idx1-ubyte');
    s = 60000; %dataset size
    o = 10; %output layer size
    
    % Transform the labels to correct target values.
    targetValues = zeros(o,s); 
    for n = 1:s
        targetValues(labels(n) + 1, n) = 1;
    end;
    
    
    % Choose activation function.
    activationFunction = @logisticSigmoid;
    dActivationFunction = @dLogisticSigmoid;
    
    
    % plotting setup
    maxIter = s/batchSize;
    iterSelection = (maxIter/20):(maxIter/20):maxIter; %select 20 iterations to plot error for
    iterSelection = ceil([iterSelection(1)/5 iterSelection]); %add a 21st early iteration
    objValues = zeros(sims,21,2);
    
    privateVals = [0 1];
    for i = 1:2 %perform private and non-private runs
        privacy = privateVals(i);
        for sim = 1:sims
            fprintf ('Trial %i of %i\n',sim,sims);

            [~, ~, objValues(sim,:,i)] = trainStochasticSquaredErrorTwoLayerPerceptron(activationFunction, dActivationFunction, u, inputValues, targetValues, maxIter, batchSize, learningRate, privacy, C, epsilon);
        end
    end
    
    % Plotting
    figure; hold on;
    if sims == 1
        plot(iterSelection,objValues(:,:,2));
        plot(iterSelection,objValues(:,:,1),'--');
        xlim([0 maxIter]);
        ylim([0 7]);
        legend('private','non-private');
        xlabel('Number of iterations');
        ylabel('Mean-squared-error');
        tit = sprintf ('d = %i, u = %i',d,u);
        title(tit);
    else
        mn = mean(objValues(:,:,1));
        pmn = mean(objValues(:,:,2));
        sd = std(objValues(:,:,2));
        errorbar(iterSelection,pmn,sd);
        plot(iterSelection,mn,'--');
        xlim([0 maxIter]);
        ylim([0 7]);
        legend('private','non-private');
        xlabel('Number of iterations');
        ylabel('Mean-squared-error');
        tit = sprintf ('d = %i, u = %i',d,u);
        title(tit);
    end
    
    hold off;
        
    
    
end