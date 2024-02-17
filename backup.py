import numpy as np
import pandas as pd 


def gradient_descent(X, y):
    #initialize the parameters to zero
    theta0 = 0  #y intercept
    theta1 = 0 #slope
    num_iters = 10
    n = len(y)
    learning_rate = 0.00000001
    loss_prev = 0
    for i in range(num_iters):
        y_pred = theta0 + theta1 * X
        #Mean Squared Error
        loss = (1/n) * sum([val**2 for val in (y - y_pred)])
        #Derivative of the loss function wrt theta0 and theta
        dtheta0 = -(2/n) * sum(y - y_pred)
        dtheta1 = -(2/n) * sum(X * (y - y_pred))
        # revised y intercept and slope
        print("theta0: {}, theta1: {}, loss: {:.2e}, prev_loss: {}".format(theta0, theta1, loss, loss-loss_prev))
        theta0 = theta0 - learning_rate * dtheta0
        theta1 = theta1 - learning_rate * dtheta1
        if abs(loss - loss_prev) < 1:
            print("Converged at iteration: ", i)
            break
        loss_prev = loss
    
    

def main():
    try:
        df = pd.read_csv("data.csv")
        y = np.array(df['km'])
        X = np.array(df['price'])
        if (X.size != y.size):
            raise ValueError("The number of rows in mileage and price columns do not match")
        # print(df)
        # gradient_descent(X, y)
    except Exception as e:
        print(type(e).__name__ + ": " + str(e))

 
    
          
if __name__ == "__main__":
    main()