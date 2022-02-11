X = [0 0 0 1; 0 0 1 1; 0 1 0 1; 0 1 1 1; 1 0 0 1; 1 0 1 1; 1 1 0 1; 1 1 1 1];
# changed expected output for sigmoid function 0 changed to 0.1 and 1 changed to 0.9
y = [0.1 0.1 0.1 0.9 0.1 0.9 0.9 0.9];


# sigmoid inverse function
function new_y = asigmoid(y)
  # create new vector
  new_y = []; 
  # loop for every output
  for index = 1:(columns(y))
    # calculate X * w
    A = -log(1/y(index)-1);
    # append to the new vektor
    new_y = [new_y A];
  endfor
endfunction

# calculate inverse outputs
asig_Y = asigmoid(y);
# calculate weights
w = X\asig_Y';
disp(w);

#display output for every input
for index = 1:rows(X)
    x = X(index, :);
    hypothesis = sigmoid(dot(w,x));
    disp(hypothesis);
endfor
