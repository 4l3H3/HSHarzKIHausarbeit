from numpy import dot, array, zeros
from random import choice, seed
from math import exp

class Perceptron:
    
    def __init__(self, iterations = 20, training_data_set = [], learnrate = 0.3) -> None:
        self.iterations = iterations
        self.training_data_set = training_data_set
        self.threshhold = len(training_data_set[0][0]) / 2
        self.learnrate = learnrate
    
    def sigmoid(self, x):
        return 1/(1 + exp(-x))
    
    def heaviside(self, x):
        if x < 0:
            return 0
        else:
            return 1
    
    def fit(self, training_data_set, w):       
        for i in range(self.iterations):
            
            # choose random training data
            training_data = choice(training_data_set)
            x = training_data[0]
            y = training_data[1]
            
            # calculate dot product 
            hypothesis = self.heaviside(dot(w,x))
            # calculate error
            error = y - hypothesis
            
            if hypothesis > 0.5:
                print("Iteration:{}, X: {}, Weights: {}, y: {}, Predict:{}, Error:{}".format(i + 1, x, w, y, hypothesis, error))
            # calculate new weights based in error
            w += self.learnrate * error * x
            

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
        (array([1,1,1,1]), 1)
    ]    
    w = zeros(4) # [0.,0.,0.,0.]
    iterations = 1000

    p = Perceptron(iterations=iterations, training_data_set=training_data_set, learnrate=1)
    p.fit(training_data_set, w)




main()