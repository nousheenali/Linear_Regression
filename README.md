## LINEAR REGRESSION

LINEAR REGRESSION is a statistical method that models the relationship between two variables.
It is used to predict the value of a dependent variable based on one or more independent variables.

The difference between the actual value and the predicted value is called the **residual**.The residual is the error in the prediction.
The line that minimizes the sum of the squared residuals is called the **Best Fit Line**. 

<p align="center"><img width="600" alt="LINEAR REGRESSION" src="https://github.com/nousheenali/Linear_Regression/assets/66158938/23baf4fb-8aee-4445-b4a6-1585fb81ec97"/></p>


> [!IMPORTANT] 
>The goal of linear regression is to minimize the sum of the squared residuals for each data point.

There are various ways to calculate the cost function(loss):

        squared_residuals = sum((actual - predicted) ** 2)
        mean_squared_error = sum((actual - predicted) ** 2) / n

Squared residuals and mean squared error are used to evaluate the performance of the model. These metrics measure the difference between the actual and predicted values. The smaller the value, the better the model.

> [!NOTE]  
> Errors and Residues are completely different concepts. Errors mainly refer to difference
between actual observed sample values and your predicted values. In contrast, residues refer
exclusively to the differences between dependent variables and estimations from linear regression.



The equation of a simple linear regression model is:

        y = mx + b
        where:  y = dependent variable
                x = independent variable
                m = slope of the line
                b = y-intercept


The equation of a multiple linear regression model is:

        y = b0 + b1x1 + b2x2 + ... + bnxn
        where:  y = dependent variable
                x1, x2, ..., xn = independent variables
                b0 = y-intercept
                b1, b2, ..., bn = coefficients

        - The coefficients are the values that minimize the sum of the squared residuals.
        - The y-intercept is the value of y when all the independent variables are 0.

## Gradient Descent

Gradient Descent is an optimization algorithm used to find optimum coefficient(slope and y-intercept) values that minimize the cost function in machine learning. The cost function measures the difference between the actual and predicted values.


[<img width="600" src="https://github.com/nousheenali/Linear_Regression/assets/66158938/597aac7b-f39a-42a8-a971-9fd3eb52bada">](https://machine-learning.paperspace.com/wiki/gradient-descent)


We use gradient descent algorithm to minimize our cost-function f(m, b) and reach its local minimum by tweaking its parameters (m and b). The image above shows the horizontal axes representing the parameters (m and b), while the cost function f(m, b) is represented on the vertical axes. We start by initializing m and b with some random numbers. Gradient descent then starts at that point (somewhere around the top of our illustration), and it takes one step after another in the steepest downside direction (i.e., from the top to the bottom of the illustration) until it reaches the point where the cost function is as small as possible.

The size of the steps taken by gradient descent in the direction of the local minimum is determined by the learning rate. To ensure the gradient descent algorithm converges to the local minimum, it is crucial to set the learning rate to an appropriate value—neither too low nor too high. If the steps are excessively large, the algorithm might oscillate and fail to reach the local minimum within the convex function. Conversely, a very small learning rate will eventually lead to convergence, but the process may be time-consuming. Finding a balanced learning rate is essential for the effectiveness of the gradient descent optimization.



### Steps to find optimal coefficients using gradient descent:

1. Start with initial values for the slope (m) and intercept (b). 
2. Use the current values of m and b to calculate the cost function.
<p align="center">
        <img width="250" alt="MSE" src="https://github.com/nousheenali/Linear_Regression/assets/66158938/db7000a0-f912-441d-bf23-5983e0f4dcbc">
        <p align="center">where ŷ is the predicted value. The above formula can therefore be re-written as:</p>
</p>
<p align="center">
<img width="300" alt="function with parameters" src="https://github.com/nousheenali/Linear_Regression/assets/66158938/4754e2fb-5011-426f-ac5e-35fb0902e09c">
</p>

3. Calculate the gradient of the cost function with respect to each parameter (m and b). The gradient points in the direction of the steepest increase of the cost function.

        
<p align="center">
        <text>To find how each parameter affects the MSE, we need to find the partial derivatives with respect to m and b</text>
<img width="712" alt="partial derivatives" src="https://github.com/nousheenali/Linear_Regression/assets/66158938/2a685a80-2f15-4b2a-9209-479cf5c4d58a">
</p>

> [!NOTE]  
> Partial Derivatives gives the instantanious slope of a point on a curve. Each derived function can tell which way we should tune parameters and by how much.

4. Update the parameters(m and b) by iterating through our derived functions and gradually minimizing MSE. In this process, we use an additional parameter **learning rate** which helps us define the step we take towards updating parameters with each iteration. 

        m = m - learning_rate * ∂f/∂m

        b = b - learning_rate * ∂f/∂b


5. Repeat steps 2, 3 and 4 until convergence or a predetermined number of iterations. Convergence is typically determined by observing a small change in the cost function or when the algorithm reaches a specified number of iterations.

## Coefficient of Determination or R-squared

The coefficient of determination (R-squared) is a measure of how well the model fits the data.

If R-squared is 1, the model perfectly predicts the dependent variable.

If R-squared is 0, the model does not predict the dependent variable at all.

[<img width="600" alt="Screen Shot 2024-02-18 at 1 31 39 AM" src="https://github.com/nousheenali/Linear_Regression/assets/66158938/75bb6433-c441-458e-88a5-4b0bd972b8f9"/>](https://www.analyticsvidhya.com/blog/2021/10/everything-you-need-to-know-about-linear-regression/)

formula for R-squared:
        
        R-squared = 1 - (sum of squared residuals / sum of squared differences from the mean)
        
        sum of squared residuals = sum((actual - predicted) ** 2)
        sum of squared differences from the mean = sum((actual - mean) ** 2)


#### HOW GRADIENT DESCENT ALGORITHM ITERATES THROUGH DIFFERENT COEFFICIENT VALUES TO FIND BEST FIT LINE

![GradientDescent](https://github.com/nousheenali/Linear_Regression/assets/66158938/8ed212f8-0395-4acb-b1cb-19a72b992317)



REFERENCES:
- [Everything you need to Know about Linear Regression!](https://www.analyticsvidhya.com/blog/2021/10/everything-you-need-to-know-about-linear-regression/)
- [Linear Regression](https://www.youtube.com/watch?v=nk2CQITm_eo)
- [Gradient Descent and Cost Function](https://www.youtube.com/watch?v=vsWrXfO3wWw)
- [Gradient Descent From Scratch](https://towardsdatascience.com/gradient-descent-from-scratch-e8b75fa986cc)
- [Gradient Descent in Machine Learning](https://builtin.com/data-science/gradient-descent)
- [Derivative as a Concept](https://www.khanacademy.org/math/ap-calculus-ab/ab-differentiation-1-new/ab-2-1/v/derivative-as-a-concept)
- [How to Calculate R squared in Linear Regression](https://www.shiksha.com/online-courses/articles/how-to-calculate-r-squared-in-linear-regression/#What-is-R-squared?)


