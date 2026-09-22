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
            x_data, y_data, initial_params,
            x_error = None, y_error = None,
            model: function = straight_line_model, diff: function = straight_line_diff,
            fmin: function = minimise,
            method: str = 'ols',
            printer: bool=True) -> None:
        # Raw data stored in global variables 
        self.raw_dataset: dict[str] = {'x_data': x_data,
                                       'y_data': y_data,
                                       'x_error': x_error,
                                       'y_error': y_error,
                                       'params': initial_params}
        self.model: function = model
        self.diff: function = diff
        self.fmin: function = fmin
        self.method: str = method
        self.printer: bool = printer
        

        # Create variables to be used for fitting
        self.dataset: dict[str, np.ndarray[float] | None] = {'x_data': None,
                                                             'y_data': None,
                                                             'x_error': None,
                                                             'y_error': None,
                                                             'params': None}
    def _convert_data_to_arrays(self) -> bool:
        '''
        Self explanatory converting the data entries into numpy arrays 
        np.asarray used as it doesnt put arrays into more arrays
        WARNING Forces matrices into np.ndarray type if this becomes a problem later np.asanyarray preserves subclasses
        True means it failed
        If errors are none before converting returns a nan value BUT the raw dataset remains the same remember
        '''
        for data in self.dataset:
            try:
                self.dataset[data] = np.asarray(self.raw_dataset[data], dtype=float)
            except (ValueError, TypeError) as e:
                if self.printer:
                    print(f'ERROR: {data} must contain only numbers - {e}')
                return True
        return False
    def _convert_errors(self) -> None:
        errors = ['x_error', 'y_error']
        for error in errors:
            if self.raw_dataset[error] is None:
                self.dataset[error] = np.zeros_like(self.dataset['x_data'])
            elif np.isscalar(self.raw_dataset):
                self.dataset[error] = np.full_like(self.dataset['x_data'], self.raw_dataset[error], dtype=float)
    def _length_check(self) -> bool:
        '''
        Check the length of the arrays matches each others
        '''
        if not (len(self.dataset['x_data']) == len(self.dataset['y_data']) == len(self.dataset['x_error']) == len(self.dataset['y_error'])):
            return True
        return False


x=[1,2,3]
y=[2,4,6]
xerr=1
yerr=None
param=[0,2]
fit=Fit(x,y,param,xerr,yerr)
print(fit._convert_data_to_arrays())
print(1 if fit.dataset['y_error'] is None else 0)
