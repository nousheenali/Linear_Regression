## LINEAR REGRESSION

**LINEAR REGRESSION** is a statistical method that models the relationship between two variables.
It is used to predict the value of a dependent variable based on one or more independent variables.

The difference between the actual value and the predicted value is called the **residual**.
The residual is the error in the prediction.

        residual = actual - predicted

> [!IMPORTANT] 
>The goal of linear regression is to minimize the sum of the squared residuals for each data point.
The line that minimizes the sum of the squared residuals is called the line of best fit.

        squared_residuals = sum((actual - predicted) ** 2)
        mean_squared_error = sum((actual - predicted) ** 2) / n

Squared residuals and mean squared error are used to evaluate the performance of the model.
These metrics measure the difference between the actual and predicted values. 
The smaller the value, the better the model.

> [!NOTE]  
> Errors and Residues are completely different concepts. Errors mainly refer to difference
between actual observed sample values and your predicted values. In contrast, residues refer
exclusively to the differences between dependent variables and estimations from linear regression.



        The equation of a simple linear regression model is:

        y = mx + c
        where:  y = dependent variable
                x = independent variable
                m = slope of the line
                c = y-intercept

        The equation of a multiple linear regression model is:

        y = b0 + b1x1 + b2x2 + ... + bnxn
        where:  y = dependent variable
                x1, x2, ..., xn = independent variables
                b0 = y-intercept
                b1, b2, ..., bn = coefficients

        - The coefficients are the values that minimize the sum of the squared residuals.
        * The y-intercept is the value of y when all the independent variables are 0.

## Gradient Descent

Gradient Descent is an optimization algorithm used to minimize the cost function in machine learning.
The cost function measures the difference between the actual and predicted values.
The goal of gradient descent is to find the values of the coefficients that minimize the cost function.

- The algorithm starts with initial values for the coefficients. In our case we set it to zero.
    slope    m = 0
    y-intercept    c = 0
- Calclate the 





The coefficient of determination (R-squared) is a measure of how well the model fits the data.
It is the proportion of the variance in the dependent variable that is predictable from the independent variables.
If R-squared is 1, the model perfectly predicts the dependent variable.
If R-squared is 0, the model does not predict the dependent variable at all.

formula for R-squared:
R-squared = 1 - (sum of squared residuals / sum of squared differences from the mean)
sum of squared residuals = sum((actual - predicted) ** 2)
sum of squared differences from the mean = sum((actual - mean) ** 2)

p-value: A p-value is the probability that the null hypothesis is true.
null hypothesis: The null hypothesis is a general statement that there is no relationship between two measured phenomena.
If the p-value is less than the significance level (usually 0.05), the null hypothesis is rejected.
If the p-value is greater than the significance level, the null hypothesis is not rejected.

The p-value for each independent variable tests the null hypothesis that the variable has no effect on the dependent variable.
If the p-value is less than the significance level, the variable has a significant effect on the dependent variable.

Reference: https://www.youtube.com/watch?v=nk2CQITm_eo


REFERENCES:
https://www.analyticsvidhya.com/blog/2021/10/everything-you-need-to-know-about-linear-regression/
"""
