import numpy as np
import matplotlib.pyplot as plt
import random

#Note: Form is wx + b = y
# NOTE: 2.7x + 0 = y is the equation we are trying to learn
# so w = 2 and b = 5

# Generate 50 training examples
# Training examples are generated with some randomness around this line
trainExamples = []
for i in range(50):
    # Generate s with some randomness between -10 and 10
    x = random.uniform(-10, 10)
    # Calculate y using the equation of the line, but add some randomness to it
    y = 2.7 * x + random.uniform(-1, 1)
    trainExamples.append((x, y))

def plotTrainExamples():
    # Plots the training examples
    plt.scatter([x for x, y in trainExamples], [y for x, y in trainExamples])
    plt.show()

#This is a feature vector that represents a linear equation of the form y = wx + b,
def phi(x):
    return np.array([1,x])

def initialWeightVector():
    # Random initial weight vector 0x + 0 = y
    return np.zeros(2)

def trainLoss(w):
    # This function calculates the mean squared error loss for the current model, given by the weight vector w. 
    # This is the function we want to minimize.
    return 1.0 / len(trainExamples) * sum(2 * (w.dot(phi(x)) - y)**2 for x, y in trainExamples)

def gradientTrainLoss(w):
    # This function calculates the gradient of the loss function at the current model, given by the weight vector w.
    # The gradient is calculated using the chain rule.
    return 1.0 / len(trainExamples) * sum(2 * (w.dot(phi(x)) - y) * phi(x) for x, y, in trainExamples)

def gradientDescent(F, gradientF):
    w = initialWeightVector()
    eta = 0.001  # learning rate
    loss_values = []  # to store the loss values for each epoch
    t = 0  # counter for the epochs
    for t in range(100):  # runs for 100 epochs
        value = F(w)  # calculates the loss value
        gradient = gradientF(w)  # calculates the gradient
        w = w - eta * gradient  # updates the weights using the gradient
        loss_values.append(value)  # stores the loss value

        plt.clf()  # clear the current figure
        # plot the training examples
        plt.scatter([x for x, y in trainExamples], [y for x, y in trainExamples])
        # plot the current regression line
        x_values = np.linspace(-10, 10, 100)
        y_values = w[1] * x_values + w[0]
        plt.plot(x_values, y_values, 'r')
        plt.draw()  # draw the plot
        plt.pause(0.1)  # pause for a short period to allow the plot to update

        print(f'epoch{t}: w (i.e. current weight) = {w}, loss (mean squared error) = {value}, gradient = {gradient}')

        # wait for the user to press enter before continuing
        input("Press enter to continue to the next epoch...")


def main():
    # Plots the training examples and performs gradient descent to learn the model
    plt.ion()  # turn on interactive mode
    gradientDescent(trainLoss, gradientTrainLoss)

if __name__ == "__main__":
   main()
