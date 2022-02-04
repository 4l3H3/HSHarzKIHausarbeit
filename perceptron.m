X = [0 0 0 1; 0 0 1 1; 0 1 0 1; 0 1 1 1; 1 0 0 1; 1 0 1 1; 1 1 0 1; 1 1 1 1];
y = [0 0 0 1 0 1 1 1];

# sigmoid activation function
function y = sigmoid(x)
  y = 1 / (1 + exp(-x));
endfunction

# step function
function y = heaviside(x)
  if (x < 0)
    y = 0;
  else 
    y = 1;
  endif
endfunction

# fitting function ; X = input, y = expected output, returns weights
function w = fit(X, y, loops = 100, learningrate = 1)
  # initialize weights based on input length
  length = columns(X);
  w = zeros(1,length);
  
  # loop through the inputs
  for index = 1:rows(X)
    # pick current input and expected output
    x = X(index, :); 
    expected = y(index);
    # predict the output based on current weights
    hypothesis = sigmoid(dot(w,x));
    # calculate error
    error = expected - hypothesis;
    # perceptron learing rule
    w += learningrate * error * x;
  endfor
endfunction



final_weights = fit(X,y);
disp(final_weights);
