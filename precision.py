import numpy as np
from utils import estimate_price, denormalize_data


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
