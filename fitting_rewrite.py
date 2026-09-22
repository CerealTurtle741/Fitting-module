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
from typing import Any, Callable

def straight_line_model(params, x_data) -> np.ndarray:
    '''
    y = intercept + x_data * gradient
    '''
    return params[0] + x_data * params[1]

def straight_line_diff(params, x_data) -> float:
    '''
    dy/dx = gradient
    '''
    return params[1]

def minimise(params, x_data, y_data, x_error, y_error, model, diff) -> np.ndarray:
    '''
    Calculating the residuals of the data (vertical distance between the data point and its predicted value)
    Used for least squares fitting 
    '''
    y_model: np.ndarray = model(params, x_data)
    y_model_diff: float | np.ndarray = diff(params, x_data)
    # x error is included by weighting it with the derivative of the model
    y_diff_total: np.ndarray = np.sqrt(y_error**2 + x_error**2 * y_model_diff**2)
    # if y_diff_total is 0 maybe from no errors recorded replace it with one to stop divide by zero errors
    y_diff_total = np.where(y_diff_total == 0, 1, y_diff_total)
    return (y_data - y_model) / y_diff_total



class Fit:
    def __init__(
            self,
            x_data, y_data, initial_params,
            x_error = None, y_error = None,
            model: Callable = straight_line_model, diff: Callable = straight_line_diff,
            fmin: Callable = minimise,
            method: str = 'ols',
            printer: bool=True) -> None:
        # Raw data stored in global variables 
        self.raw_dataset: dict[str, Any] = {'x_data': x_data,
                            'y_data': y_data,
                            'x_error': x_error,
                            'y_error': y_error,
                            'params': initial_params}
        self.model: Callable = model
        self.diff: Callable = diff
        self.fmin: Callable = fmin
        self.method: str = method
        self.printer: bool = printer
        self.data_arrays: list[str] = ['x_data', 'y_data', 'x_error', 'y_error']
        self.errors: list[str] = ['x_error', 'y_error']


        

        # Create variables to be used for fitting
        self.dataset: dict[str, Any] = {'x_data': None,
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
        '''
        convert errors into arrays of zeros if none or arrays full of a value if the only have one value
        '''
        for error in self.errors:
            if self.raw_dataset[error] is None:
                self.dataset[error] = np.zeros_like(self.dataset['x_data'])
            elif np.isscalar(self.raw_dataset[error]):
                self.dataset[error] = np.full_like(self.dataset['x_data'], self.raw_dataset[error], dtype=float)
    def _length_check(self) -> bool:
        '''
        Check the length of the arrays matches each others
        '''
        if not (len(self.dataset['x_data']) == len(self.dataset['y_data']) == len(self.dataset['x_error']) == len(self.dataset['y_error'])):
            if self.printer:
                print('ERROR: Data arrays not the same length')
                for data in self.data_arrays:
                    print(f'Length of {data} array: {len(self.dataset[data])}')
            return True
        return False
    def _parameter_check(self) -> bool:
        '''
        making sure there is enough data points for the number of parameters
        e.g. 3 for a linear relationship
        '''
        self.nparams = len(self.dataset['params'])
        if len(self.dataset['x_data']) < self.nparams + 1:
            if self.printer:
                print('ERROR: number of data points less the number of parameters')
            return True
        return False
    def _error_negativity_check(self) -> bool:
        '''
        the errors cannot be negative otherwise it will break the fitting models 
        '''
        for error in self.errors:
            if self.dataset[error] < 0:
                if self.printer:
                    print(f'ERROR: {error} must not have negative values')
                return True
        return False
    def _model_check(self) -> bool:
        try:
            self.model(self.dataset['params'], self.dataset['x_data'])
        except Exception as e:
            if self.printer:
                print(f'ERROR: Model received incorrect number of parameters - {e}')
            return True
        return False


