function [ inputValues , targetValues ] = loadMNIST(d)
% LOADMNIST load the MNIST dataset, returning the input values and the 0 vs
% the rest classification target values
%

    inputValues1 = loadMNISTImages('train-images.idx3-ubyte');
    targetValues = loadMNISTLabels('train-labels.idx1-ubyte');

    for i = 1:size(targetValues,1)
        if targetValues(i) == 0
            targetValues(i) = 1;
        else
            targetValues(i) = -1;
        end
    end

    if d < 784
        % generate random projection matrix with normalized rows
        R = rand(d,784);
        for i = 1:d
            v = R(i,:);
            R(i,:) = v / max(1,norm(v));
        end

        % reduce dimension size by random projection
        inputValues = R*inputValues1;
    else %meant for d = 784, the original input dimension size
        inputValues = inputValues1;
    end

end