import numpy as np
import pandas as pd 


def gradient_descent(X, y, theta0, theta1, num_iters):
    m = len(y)
    for i in range(num_iters):
        y_pred = theta0 + theta1 * X
        cost = (1 / (2 * m)) * np.sum((y_pred - y) ** 2)
        d_theta0 = (1 / m) * np.sum(y_pred - y)
        d_theta1 = (1 / m) * np.sum((y_pred - y) * X)
        theta0 = theta0 - 0.01 * d_theta0
        theta1 = theta1 - 0.01 * d_theta1
        print(f"theta0: {theta0} | theta1: {theta1} | cost: {cost} | iteration: {i}


    


def main():
    df = pd.read_csv("data.csv")
    y = df.drop('km', axis=1)
    X = df.drop('price', axis=1)
    theta0 = 0
    theta1 = 0
    gradient_descent(X, y, theta0, theta1, 1000)

 
    
          
if __name__ == "__main__":
    main()