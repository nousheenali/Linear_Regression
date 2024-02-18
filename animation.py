import matplotlib.pyplot as plt
from utils import estimate_price
from matplotlib.animation import FuncAnimation  # for creating animated plots in Matplotlib

"""
Use this function to have an animated plot of the gradient descent process.
"""

def gradient_descent_animation(x, y):
    """
    Train the model using gradient descent to get the optimal values of
    theta0 and theta1 by minimizing the loss function (Mean Squared Error
    in this case)
    """
    # Initializing values
    theta0 = 0  # y-intercept
    theta1 = 0  # slope
    num_iters = 100  # number of iterations
    m = len(y)  # number of data points
    learning_rate = 0.9  # hyperparameter

    fig, ax = plt.subplots()
    ax.scatter(x, y, color='red')  # Plot normalised data points
    ax.set_xlabel('Normalised Mileage')
    ax.set_ylabel('Normalised Price')
    ax.set_title('Best Fit lines for different Gradient Descent iterations', fontsize=10)
    # create an empty green line that will be updated in the animation
    # (Refer to NOTE1 for explanation of parameters)
    line, = ax.plot([], [], color='green', lw=2)  # lw - line width

    #  Initializes the plot with an empty line.
    def init():
        line.set_data([], [])
        return line,

    def update(frame):
        """
        Updates the plot for each iteration of gradient descent.
        It calculates the predicted values, updates the green line,
        and adjusts the parameters theta0 and theta1.
        """
        nonlocal theta0, theta1

        y_pred = estimate_price(x, theta0, theta1)
        loss = (1/m) * sum([val**2 for val in (y - y_pred)])

        # update the data of a Line2D object in Matplotlib
        line.set_data(x, y_pred)

        tmp_theta0 = learning_rate * 1/m * sum(y_pred - y)
        tmp_theta1 = learning_rate * 1/m * sum((y_pred - y) * x)

        theta0 = theta0 - tmp_theta0
        theta1 = theta1 - tmp_theta1

        print("Iteration: {}, theta0: {:.4f}, theta1: {:.4f}, loss: {:.4f}".format(frame, theta0, theta1, loss))

        return line,

    # Create the animation(refers to NOTE2 for explanation of parameters)
    ani = FuncAnimation(fig, update, frames=num_iters, init_func=init, blit=True)
    plt.show()

    print("Final values - theta0: {:.4f}, theta1: {:.4f}, loss: {:.4f}".format(theta0, theta1, loss))
    return theta0, theta1


"""
NOTE1:
ax.plot([], [], color='green', lw=2): This creates a line plot with initially
empty data, specified color, and linewidth. The result is a list containing one
Line2D object.

line, = ax.plot([], [], color='green', lw=2): The comma after line is used for
unpacking. It says "take the first element of the list returned by ax.plot and
assign it to the variable line." If you were to omit the comma and write
line = ax.plot([], [], color='green', lw=2), line would be assigned a list
containing the Line2D object, and later when you try to use line.set_data(...),
it would result in an error because line would be a list, not a Line2D object.
It's a common Python idiom for unpacking a single-element tuple or list.

example of Line2D object:
line = Line2D([0, 1, 2], [0, 1, 0], color='blue', linestyle='-', linewidth=2, marker='o', markersize=8)

NOTE2:
ani = FuncAnimation(fig, update, frames=num_iters, init_func=init, blit=True)
fig - The figure to be animated
Update - The function that is called at each frame of the animation
frames - The number of frames to be created
init_func - The function used to draw a clear frame
blit - Whether blitting is used to optimize drawing. 
If blit=True, only the parts of the figure that have changed are redrawn.


"""
