import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from precision import r_sqaured
from utils import (
    estimate_price,
    denormalize_data,
    normalize_data,
    plot_graph,
    print_result
)
from animation import gradient_descent_animation


def gradient_descent(x, y, ax0):
    """
    Train the model using gradient descent to get the optimal values of
    theta0 and theta1 by minimizing the loss function (Mean Squared Error
    in this case)
    """
    # Initializing values
    theta0 = 0  # y-intercept
    theta1 = 0  # slope
    num_iters = 1000  # number of iterations
    m = len(y)  # number of data points
    learning_rate = 0.9  # hyperparameter

    ax0.scatter(x, y, color='red')  # Plot normalised data points

    # plt.scatter(x, y, color='red')  # Plot normalised data points

    # Gradient Descent iterations for different values of theta0 and theta1
    for i in range(num_iters):

        y_pred = estimate_price(x, theta0, theta1)
        loss = (1/m) * sum([val**2 for val in (y - y_pred)])

        # Plot the line for each iteration
        ax0.plot(x, y_pred, color='green')

        # Derivative of the loss function wrt theta0 and theta1
        # multiplied by the learning rate
        tmp_theta0 = learning_rate * 1/m * sum(y_pred - y)
        tmp_theta1 = learning_rate * 1/m * sum((y_pred - y) * x)

        # Update theta0 and theta1 for new line
        theta0 = theta0 - tmp_theta0
        theta1 = theta1 - tmp_theta1

    ax0.set_xlabel('Normalised Mileage')
    ax0.set_ylabel('Normalised Price')
    ax0.set_title(
        'Best Fit lines for different Gradient Descent iterations',
        fontsize=10)
    ax0.legend(['Data points', 'Best fit lines'])
    print("theta0: {}, theta1: {}, loss: {}".format(theta0, theta1, loss))
    print("--------------------------------------------")
    return theta0, theta1


def train_model(path_to_csv_file, mil):
    """
    Train the model using the data from the csv file and
    estimate the price of a car given its mileage
    """
    try:
        # Loading the data from the csv file
        data = pd.read_csv(path_to_csv_file)
        if data is None:
            raise Exception('Unable to load data.csv')

        # Spearte dependent(price) and independent(mileage) variables
        X = np.array(data['km'])
        y = np.array(data['price'])
        if X.size != y.size:
            raise ValueError(
                "The number of rows in mileage and price columns do not match")

        # Normalizing the data to be between 0 and 1
        # This step is important for gradient descent to converge
        X_norm = normalize_data(X, X)
        y_norm = normalize_data(y, y)

        # setting up the subplots
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        # Training model using gradient descent to get optimal theta0 & theta1
        theta0, theta1 = gradient_descent(X_norm, y_norm, ax[0])
        # for animated gradient descent  refer NOTE1
        # theta0, theta1 = gradient_descent_animation(X_norm, y_norm)

        # Normalizing the mileage value to be between 0 and 1
        normalised_mil = normalize_data(mil, X)
        price = estimate_price(normalised_mil, theta0, theta1)

        # Denormalizing the price to get the actual price
        denormalised_price = denormalize_data(price, y)
        print_result(denormalised_price)

        # r-squared
        r_sqaured(X, y, theta0, theta1, X_norm)

        # plot the graph
        plot_graph(X_norm, theta0, theta1, X, y, ax[1])
        plt.show()

    except Exception as e:
        print(type(e).__name__ + ": " + str(e))
        exit(1)
    except KeyboardInterrupt:
        print("Exiting...")
        exit(1)


"""
NOTE1:
For animated gradient descent, use the following code:
    theta0, theta1 = gradient_descent_animation(X_norm, y_norm)

    When using this comment out the following LINES in the above code:
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    theta0, theta1 = gradient_descent(X_norm, y_norm, ax[0])
    plot_graph(X_norm, theta0, theta1, X, y, ax[1])
"""
