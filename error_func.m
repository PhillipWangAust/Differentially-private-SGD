function [ f , g ] = error_func( x , input , target , lambda )
%error_func Evaluation of the error function and its gradient
%f(w,x,y,la) = (la/2)*norm(w)^2 + (1/n)Sum[i=1 to n]log(1+exp(-y*w'*x))
%f'(w,x,y,la) = la*x-y*x/(1+exp(y*w'*x))

    %% function evaluation
    n = size(input,2);

    loss =  log( 1 + exp(  -target' .* (x' * input) ) );

    f = (lambda/2)*(x'*x) + sum(loss)/n;

    %% gradient evalutation
    if ( nargout > 1 )

        dlsum = 0;

        for i = 1:n
            ywx = target(i) * x'*input(:,i);

            dlsum = dlsum + ( -target(i) ./ (1 + exp( ywx )) ) .* input(:,i);
        end

        g = lambda*x + dlsum/n;
    end

end

