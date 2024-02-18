
import matplotlib.pyplot as plt

# default values for theta0 and theta1 are 0
def estimate_price(mileage, theta0=0, theta1=0):
    """
    Estimate the price of a car given its mileage
    The formula represents a straight line: y = theta0 + (theta1 * x)
    """
    return theta0 + (theta1 * mileage)


def normalize_data(value, data):
    """Normalize the data to be between 0 and 1"""
    return (value - data.min()) / (data.max() - data.min())


def denormalize_data(value, data):
    """Denormalize the data to get the actual value"""
    return (value * (data.max() - data.min())) + data.min()


def plot_graph(X_norm, theta0, theta1, X, y, ax1):
    """Plot the graph with the best fit line"""
    ax1.scatter(X, y, color='red')
    y_pred = estimate_price(X_norm, theta0, theta1)
    y_denomralized = denormalize_data(y_pred, y)
    ax1.plot(X, y_denomralized, color='blue')
    ax1.set_xlabel('Mileage')
    ax1.set_ylabel('Price')
    ax1.set_title('Best Fit Line', fontsize=10)
    ax1.legend(['Data points', 'Best fit line'])


def print_result(price):
    """Print the estimated price of the car"""
    # After a certain mileage, price will be zero
    if price < 0:
        price = 0
    print("Estimated price from trained model: {:.2f}".format(price))
    print("--------------------------------------------")