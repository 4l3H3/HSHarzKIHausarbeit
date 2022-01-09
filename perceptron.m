X = [0 0 0 1; 0 0 1 1; 0 1 0 1; 0 1 1 1; 1 0 0 1; 1 0 1 1; 1 1 0 1; 1 1 1 1];
y = [0 0 0 1 0 1 1 1];

function y = major(value)
  if (value > 3/2)
    y = 1;
  else
    y = 0;
  endif
endfunction

function w = fit(X, y, loops = 30)
  w = zeros(1,4);
  for index = 1:loops
    random = randi(8);
    x = X(random, :);
    expected = y(random);
    hypothesis = major(dot(w,x));
    error = expected - hypothesis;
    w = w + error * x;
  endfor
endfunction



final_weights = fit(X,y);
disp(final_weights);