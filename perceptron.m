X = [0 0 0 1; 0 0 1 1; 0 1 0 1; 0 1 1 1; 1 0 0 1; 1 0 1 1; 1 1 0 1; 1 1 1 1];
y = [0 0 0 1 0 1 1 1];

function y = sigmoid(x) #definition für vektoren
  y = 1 / (1 + exp(-x));
endfunction

function w = fit(X, y, loops = 100, learnrate = 0.2)
  length = columns(X);
  w = zeros(1,length);
  for index = 1:loops
    random = randi(8);
    x = X(random, :);
    expected = y(random);
    hypothesis = sigmoid(dot(w,x));
    #stufenfunktion 
    error = expected - hypothesis;
    w += learnrate * error * x;
  endfor
endfunction



final_weights = fit(X,y);
disp(final_weights);