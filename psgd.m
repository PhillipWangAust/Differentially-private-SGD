function [ w , pw , ws , pws ] = psgd( w, inputValues, targetValues, learningRate, lambda, batchSize, epsilon0, epochs )
% psgd Implementation of SGD for solving a specific classification problem.
% Runs a private and non-private version parallel to eachother.
%
% INPUT:
% w                              : Algorithm starting point
% inputValues                    : Input values 
% targetValues                   : Classification target values
% learningRate                   : Learning rate multiplier
% lambda                         : Regularization parameter
% batchSize                      : Desired global sensitivity of each step
% epsilon0                       : Privacy budget
% epochs                         : Number of epochs after which to stop
%
% OUTPUT:
% w                              : Non-private weights
% pw                             : Private weights
% ws                             : Values of w in each iteration
% pw                             : Values of pw in each iteration
% 
    %% setup
    s = size(inputValues,2); %dataset size
    d = size(inputValues,1); %input dimension size
    maxIter = s/batchSize; %iterations per epoch
    % for storing the iterates
    ws = zeros(d,epochs*maxIter);
    pw = w; %initialize starting point for private algorithm
    pws = zeros(d,epochs*maxIter);

    % scaling for the privacy factor for epochs>1
    scales = ones(epochs,1);
    r = 1;
    for e = 1:epochs-1
        scales(e) = 2*r/(epochs+2-e);
        r = r-scales(e);
    end
    scales(epochs) = r;

    %% SGD algorithm
    for e = 1:epochs

        % apply epsilon scale
        epsilon = epsilon0 * scales(e); %scaled
        %epsilon = epsilon0 / epochs; %constant

        % randomly choose the sampling order
        samples = randperm(s);

        for t = 1:maxIter

            % select next batch in line
            batch = zeros(d,batchSize);
            batchTargets = zeros(batchSize,1);
            for k = 1:batchSize
                sample = (t-1)*batchSize+k;
                i = samples(sample); 
                batch(:,k) = inputValues(:,i);
                batchTargets(k) = targetValues(i,1);
            end

            % calculate gradients
            [~,step] = error_func(w,batch,batchTargets,lambda);
            [~,pStep] = error_func(pw,batch,batchTargets,lambda);

            % calculate step length
            eta = learningRate/(sqrt((e-1)*maxIter + t));

            % add noise to private gradient
            delta = 2;
            Z = generateNoise(d,1,delta/epsilon);
            pStep = pStep + Z/batchSize;

            % perform step
            w = w - eta*step;
            pw = pw - eta*pStep;

            % project iterate on sphere with radius 1/lambda
            nrm = norm(w);
            if nrm > 1/lambda
                w = w/(nrm*lambda);
            end
            nrm = norm(pw);
            if nrm > 1/lambda
                pw = pw/(nrm*lambda);
            end

            %% store for plotting
            ws(:,((e-1)*maxIter + t)) = w;
            pws(:,((e-1)*maxIter + t)) = pw;
        end
    end
end

