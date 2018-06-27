function [ Z ] = generateNoise( d, s, beta )
%generateNoise Noise generation steps from Wu et. al. 2017
    
    %% generate noise
    V = normrnd(0,1,d,s);

    % normalise columns of V
    for i = 1:s
        nrm = norm(V(:,i));
        V(:,i) = V(:,i)/max(1,nrm);
    end

    l = gamrnd(d*s,beta);

    Z = l*V;

end

