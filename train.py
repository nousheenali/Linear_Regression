import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# default values for theta0 and theta1 are 0
def estimate_price(mileage, theta0=0, theta1=0):
    """
    Estimate the price of a car given its mileage
    The formula represents a straight line: y = theta0 + (theta1 * x)
    """
    return theta0 + (theta1 * mileage)


def gradient_descent(x, y):
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
    # plt.scatter(x, y, color='red')  # Plot normalised data points

    # Gradient Descent iterations for different values of theta0 and theta1
    for i in range(num_iters):

        y_pred = estimate_price(x, theta0, theta1)
        loss = (1/m) * sum([val**2 for val in (y - y_pred)])

        # Plot the line for each iteration
        # plt.plot(x, y_pred, color='green')

        # Derivative of the loss function wrt theta0 and theta1
        # multiplied by the learning rate
        tmp_theta0 = learning_rate * 1/m * sum(y_pred - y)
        tmp_theta1 = learning_rate * 1/m * sum((y_pred - y) * x)

        # Update theta0 and theta1 for new line
        theta0 = theta0 - tmp_theta0
        theta1 = theta1 - tmp_theta1

    print("theta0: {}, theta1: {}, loss: {}".format(theta0, theta1, loss))
    print("--------------------------------------------")
    return theta0, theta1


def normalize_data(value, data):
    """Normalize the data to be between 0 and 1"""
    return (value - data.min()) / (data.max() - data.min())


def denormalize_data(value, data):
    """Denormalize the data to get the actual value"""
    return (value * (data.max() - data.min())) + data.min()


def plot_graph(X_norm, theta0, theta1, X, y):
    """Plot the graph with the best fit line"""
    plt.scatter(X, y, color='red')
    y_pred = estimate_price(X_norm, theta0, theta1)
    y_denomralized = denormalize_data(y_pred, y)
    plt.plot(X, y_denomralized, color='blue')
    plt.xlabel('Mileage')
    plt.ylabel('Price')
    plt.title('Best fit line')
    plt.show()


def print_result(price):
    """Print the estimated price of the car"""
    # After a certain mileage, price will be zero
    if price < 0:
        price = 0
    print("Estimated price from trained model: {:.2f}".format(price))
    print("--------------------------------------------")


def r_sqaured(X, y, theta0, theta1, X_norm):
    """Calculate the R-squared value/Coefficient of determination
    formula: 1 - (SSR/SST)
    SSR: sum of squared residuals (sum of squared errors)
    SST: total sum of squares (sum of squared deviations from the mean)
    """
    y_mean = np.mean(y)
    y_pred = estimate_price(X_norm, theta0, theta1)
    denorm_y_pred = denormalize_data(y_pred, y)
    ssr = sum((y - denorm_y_pred) ** 2)
    sst = sum((y - y_mean) ** 2)
    r_square = 1 - (ssr / sst)
    print("R-square: ", r_square)
    print("--------------------------------------------")


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

        # Training model using gradient descent to get optimal theta0 & theta1
        theta0, theta1 = gradient_descent(X_norm, y_norm)

        # Normalizing the mileage value to be between 0 and 1
        normalised_mil = normalize_data(mil, X)
        price = estimate_price(normalised_mil, theta0, theta1)

        # Denormalizing the price to get the actual price
        denormalised_price = denormalize_data(price, y)
        print_result(denormalised_price)

        # r-squared
        r_sqaured(X, y, theta0, theta1, X_norm)

        # plot the graph
        plot_graph(X_norm, theta0, theta1, X, y)

    except Exception as e:
        print(type(e).__name__ + ": " + str(e))
        exit(1)
    except KeyboardInterrupt:
        print("Exiting...")
        exit(1)
