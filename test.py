import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
data = {
    'km': [240000, 139800, 150500, 185530, 176000, 114800, 166800, 89000, 144500, 84000,
           82029, 63060, 74000, 97500, 67000, 76025, 48235, 93000, 60949, 65674,
           54000, 68500, 22899, 61789],
    'price': [3650, 3800, 4400, 4450, 5250, 5350, 5800, 5990, 5999, 6200,
              6390, 6390, 6600, 6800, 6800, 6900, 6900, 6990, 7490, 7555,
              7990, 7990, 7990, 8290]
}

df = pd.DataFrame(data)

# Feature scaling
df['km'] = df['km'] / 10000  # Feature scaling to make convergence faster

# Add a bias term (x0) to the features
df.insert(0, 'bias', 1)

# Initialize theta parameters
theta = np.zeros(2)

# Hyperparameters
alpha = 0.01
iterations = 1500

# Gradient Descent function
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    for _ in range(iterations):
        error = X @ theta - y
        gradient = X.T @ error / m
        theta -= alpha * gradient
    return theta

# Separate features and target
X = df[['bias', 'km']].values
y = df['price'].values

# Run gradient descent
theta = gradient_descent(X, y, theta, alpha, iterations)

# Display the learned parameters
print(f'Learned parameters: theta0 = {theta[0]}, theta1 = {theta[1]}')

# Plot the data and the linear regression line
plt.scatter(df['km'], df['price'], label='Original Data')
plt.plot(df['km'], X @ theta, color='red', label='Linear Regression')
plt.xlabel('Kilometers (scaled)')
plt.ylabel('Price')
plt.title('Linear Regression using Gradient Descent')
plt.legend()
plt.show()
