# Rules
## No ai to be used at all
## no copy paste from old functions
## only read the parts of the class fitting
## read full thing of function fitting 
## try to write comments about why a thing is being used 
## each function try and write a docustring of what it is doing 
## type annotations!!!!
## remember dictionaries are useful
## focus solely on fitting not plotting

import numpy as np
from scipy.optimize import least_squares
from odrpack import odr_fit
from time import time

def straight_line_model(params, x_data) -> np.ndarray[float]:
    '''
    y = intercept + x_data * gradient
    '''
    return params[0] + x_data * params[1]

def straight_line_diff(params, x_data) -> float:
    '''
    dy/dx = gradient
    '''
    return params[1]

def minimise(params, x_data, y_data, x_error, y_error, model, diff) -> np.ndarray[float]:
    '''
    Calculating the residuals of the data (vertical distance between the data point and its predicted value)
    Used for least squares fitting 
    '''
    y_model: np.ndarray[float] = model(params, x_data)
    y_model_diff: float | np.ndarray[float] = diff(params, x_data)
    # x error is included by weighting it with the derivative of the model
    y_diff_total: np.ndarray[float] = np.sqrt(y_error**2 + x_error**2 * y_model_diff**2)
    # if y_diff_total is 0 maybe from no errors recorded replace it with one to stop divide by zero errors
    y_diff_total = np.where(y_diff_total == 0, 1, y_diff_total)
    return (y_data - y_model) / y_diff_total



class Fit:
    def __init__(
            self,
            x_data, y_data,
            x_error = None, y_error = None,
            model: function = straight_line_model, diff: function = straight_line_diff,
            fmin: function = minimise,
            method: str = 'ols') -> None:
        # Raw data stored in global variables 
        self.raw_x_data = x_data
        self.raw_y_data = y_data
        self.raw_x_error = x_error
        self.raw_y_error = y_error
        self.model = model
        self.diff = diff
        self.fmin = fmin
        self.method = method