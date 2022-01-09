from numpy import dot, array, zeros
from random import choice, seed

class Perceptron:
    
    def __init__(self, iterations = 20, training_data_set = []) -> None:
        self.iterations = iterations
        self.training_data_set = training_data_set
    
    def major(self, x):
        # if the dot product is greater than the total amount of inputs, return 1
        if x > len(self.training_data_set[0][0]) / 2:
            return 1
        else:
            return 0
    
    def fit(self, training_data_set, w):       
        for i in range(self.iterations):
            
            # choose random training data
            training_data = choice(training_data_set)
            x = training_data[0]
            y = training_data[1]
            
            # calculate dot product 
            y_hat = self.major(dot(w,x))
            
            # calculate error
            error = y - y_hat
            
            print("Iteration:{}, X: {}, Weights: {}, y: {}, Predict:{}, Error:{}".format(i + 1, x, w, y, self.major(dot(x,w)), error))
            # calculate new weights based in error
            w += error * x
            


def main():
    # 3 Inputs, 1 Bias, 1 expected Output
    training_data_set = [
        (array([0,0,0,1]), 0),
        (array([0,0,1,1]), 0),
        (array([0,1,0,1]), 0),
        (array([0,1,1,1]), 1),
        (array([1,0,0,1]), 0),
        (array([1,0,1,1]), 1),
        (array([1,1,0,1]), 1),
        (array([1,1,1,1]), 1),
    ]    
    w = zeros(4) # [0.,0.,0.,0.]
    iterations = 200

    seed(12)
    p = Perceptron(iterations=iterations, training_data_set=training_data_set)
    p.fit(training_data_set, w)




main()