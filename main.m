function [] = main()
%main Solve the MNIST 0 vs the rest classification problem and plot the
%intermediary results for a selection of iterations

    % Choose hyperparameters
    batchSize = 1;
    learningRate = 1; %learning rate multiplier
    sims = 20; %amount of simulations
    regularization = 0.0001;
    epsilon = 1; %privacy budget
    epochs = 1;
    
    
    % Load dataset
    d = 15; %number of params
    s = 60000; %dataset size
    setStr = 'MNIST';
    [inputValues,targetValues] = loadMNIST(d);
    fprintf ('Solving classification for MNIST.\n');
    
    % normalize x_i
    for i = 1:s
        v = inputValues(:,i);
        inputValues(:,i) = v / max(1,norm(v));
    end
    
    
    % plotting setup
    maxIter = epochs*s/batchSize;        
    selection = (maxIter/20):(maxIter/20):maxIter; % select 20 iterations to show error for
    selection = ceil([selection(1)/5 selection]); %add a 21st early iteration
    objValues = zeros(sims,21);%non-private results
    pObjValues = zeros(sims,21); %private results

    for sim = 1:sims
        fprintf ('Trial %i of %i\n',sim,sims);


        % Initialize the parameters
        initParams = rand(d, 1)-0.5;


        % Solve the classification problem
        [ ~, ~, history, pHistory] = psgd(initParams, inputValues, targetValues, learningRate, regularization, batchSize, epsilon, epochs);

        % calculate objective values for each iteration        
        for i = 1:21
            objValues(sim,i) = error_func(history(:,selection(i)),inputValues,targetValues,regularization);
            pObjValues(sim,i) = error_func(pHistory(:,selection(i)),inputValues,targetValues,regularization);
        end
    end

    
    % Plotting
    figure; hold on;
    if sims == 1
        plot(selection,pObjValues);
        plot(selection,objValues,'--');
        xlim([0 maxIter]);
        ylim([0 3]);
        legend('private','non-private');
        xlabel('Number of iterations');
        ylabel('Value of objective');
        tit = sprintf ('%s, batch size = %i',setStr,batchSize);
        title(tit);
    else
        mn = mean(objValues);
        pmn = mean(pObjValues);
        sd = std(pObjValues);
        errorbar(selection,pmn,sd);
        plot(selection,mn,'--');
        xlim([0 maxIter]);
        ylim([0 3]);
        legend('private','non-private');
        xlabel('Number of iterations');
        ylabel('Value of objective');
        tit = sprintf ('%s, batch size = %i',setStr,batchSize);
        title(tit);
    end
    hold off;
    
end