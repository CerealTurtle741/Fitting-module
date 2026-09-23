# Rules
## No ai to be used at all
## no copy paste from old functions
## only read the parts of the class fitting
## read full thing of function fitting 
## try to write comments about why a thing is being used 
## each function try and write a docustring of what it is doing 
## type annotations!!!!
### Do not use Any as a type 
### When using assert do not use a plain assert wrap it in assert isinstance(variable, type) to prevent errors from being triggered due to truth values
## remember dictionaries are useful
## focus solely on fitting not plotting


import numpy as np
from scipy.optimize import least_squares 
from odrpack import odr_fit
from time import time
from typing import Callable
from numpy.typing import ArrayLike
from numpy import ndarray
from numpy.linalg import LinAlgError

def straight_line_model(params: ArrayLike, x_data: ArrayLike) -> ndarray:
    '''
    y = intercept + x_data * gradient
    '''
    assert isinstance(params, ndarray) and isinstance(x_data, ndarray)
    return params[0] + x_data * params[1]

def straight_line_diff(params: ArrayLike, x_data: ArrayLike) -> float:
    '''
    dy/dx = gradient
    '''
    assert isinstance(params, ndarray) and isinstance(x_data, ndarray)
    return params[1]

def minimise(params: ndarray,
             x_data: ndarray,
             y_data: ndarray,
             x_error: ndarray,
             y_error: ndarray,
             model: Callable[[ArrayLike, ArrayLike], ndarray],
             diff: Callable[[ArrayLike, ArrayLike], float | ndarray]
             ) -> ndarray:
    '''
    Calculating the residuals of the data (vertical distance between the data point and its predicted value)
    Used for least squares fitting 
    '''
    y_model: ndarray = model(params, x_data)
    y_model_diff: float | ndarray = diff(params, x_data)
    # x error is included by weighting it with the derivative of the model
    y_diff_total: ndarray = np.sqrt(y_error**2 + x_error**2 * y_model_diff**2)
    # if y_diff_total is 0 maybe from no errors recorded replace it with one to stop divide by zero errors
    y_diff_total = np.where(y_diff_total == 0, 1, y_diff_total)
    return (y_data - y_model) / y_diff_total



class Fit:
    def __init__(self,
                 x_data: ArrayLike,
                 y_data: ArrayLike,
                 initial_params: ArrayLike,
                 x_error: None | ArrayLike = None,
                 y_error: None | ArrayLike = None,
                 model: Callable[[ArrayLike, ArrayLike], ndarray] = straight_line_model,
                 diff: Callable[[ArrayLike, ArrayLike], ndarray | float] = straight_line_diff,
                 fmin: Callable[..., ndarray] = minimise,
                 printer: bool=True
                 ) -> None:
        # Raw data stored in global variables 
        self.raw_dataset: dict[str, ArrayLike | None] = {
            'x_data': x_data,
            'y_data': y_data,
            'x_error': x_error,
            'y_error': y_error,
            'params': initial_params
            }
        self.model: Callable[[ArrayLike, ArrayLike], ndarray] = model
        self.diff: Callable[[ArrayLike, ArrayLike], float | ndarray] = diff
        self.fmin: Callable[..., ndarray] = fmin
        self.printer: bool = printer


        # Create variables to be used for fitting
        self.dataset: dict[str, None | ndarray] = {
            'x_data': None,
            'y_data': None,
            'x_error': None,
            'y_error': None,
            'params': None
            }

        # Create Values to be updated after checks
        self.npoints: None | int = None
        self.nparams: None | int = None
        self.ndof: None | int = None
        self.success: None | bool = None
        self.chi2: None | float = None
        self.params: None | ndarray = None
        self.param_errors: None | ndarray = None
        self.label: None | str = None

    def run(self, method: str = 'ols'):
        start_time: float = time()
        if self._initial_checks():
            return self._failed_result()

    def _convert_data_to_arrays(self) -> bool:
        '''
        Self explanatory converting the data entries into numpy arrays 
        np.asarray used as it doesnt put arrays into more arrays
        WARNING Forces matrices into ndarray type if this becomes a problem later np.asanyarray preserves subclasses
        True means it failed
        If errors are none before converting returns a nan value BUT the raw dataset remains the same remember
        '''
        for data in ['x_data', 'y_data']:
            try:
                self.dataset[data] = np.asarray(self.raw_dataset[data], dtype=float)
            except (ValueError, TypeError) as e:
                if self.printer:
                    print(f'ERROR: {data} must contain only numbers - {e}')
                return True
        return False
    def _convert_errors_to_arrays(self) -> bool:
        '''
        convert errors into arrays of zeros if none or arrays full of a value if the only have one value
        convert errors into array if array provided
        '''
        for error in ['x_error', 'y_error']:
            if self.raw_dataset[error] is None:
                self.dataset[error] = np.zeros_like(self.dataset['x_data'])
            elif np.isscalar(self.raw_dataset[error]):
                self.dataset[error] = np.full_like(self.dataset['x_data'], self.raw_dataset[error], dtype=float)
            else:
                try:
                    self.dataset[error] = np.asarray(self.raw_dataset[error], dtype=float)
                except (ValueError, TypeError) as e:
                    if self.printer:
                        print(f'ERROR: {error} must contain only numbers - {e}')
                    return True
        return False
    def _convert_params_to_array(self) -> bool:
        '''
        convert initial parameters to an array
        '''
        try:
            self.dataset['params'] = np.asarray(self.raw_dataset['params'], dtype=float)
        except (ValueError, TypeError) as e:
            if self.printer:
                print(f'ERROR: Initial Paramters must contain only numbers - {e}')
            return True
        return False
    def _ndof_calculator(self):
        assert isinstance(self.dataset['x_data'], ndarray)
        assert isinstance(self.dataset['params'], ndarray)
        self.npoints = len(self.dataset['x_data'])
        self.nparams = len(self.dataset['params'])
        self.ndof = self.npoints - self.nparams
    def _length_check(self) -> bool:
        '''
        Check the length of the arrays matches each others
        '''
        assert isinstance(self.dataset['x_data'], ndarray)
        assert isinstance(self.dataset['y_data'], ndarray)
        assert isinstance(self.dataset['x_error'], ndarray)
        assert isinstance(self.dataset['y_error'], ndarray)
        assert isinstance(self.dataset['params'], ndarray)
        try:
            x_data_length = len(self.dataset['x_data'])
            y_data_length = len(self.dataset['y_data'])
            x_error_length = len(self.dataset['x_error'])
            y_error_length = len(self.dataset['y_error'])
        except Exception as e:
            print(f'ERROR: cannot take length of dataset - {e}')
            return True
        if not (x_data_length == y_data_length == x_error_length == y_error_length):
            if self.printer:
                print('ERROR: Data arrays not the same length')
                print(f'X Data : {x_data_length}')
                print(f'Y Data : {y_data_length}')
                print(f'X Error : {x_error_length}')
                print(f'Y Error : {y_error_length}')
            return True
        return False
    def _parameter_check(self) -> bool:
        '''
        making sure there is enough data points for the number of parameters
        e.g. 3 for a linear relationship
        '''
        assert isinstance(self.dataset['x_data'], ndarray)
        assert isinstance(self.dataset['params'], ndarray)
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
        for error in ['x_error', 'y_error']:
            if np.any(self.dataset[error]) < 0:
                if self.printer:
                    print(f'ERROR: {error} must not have negative values')
                return True
        return False
    def _model_check(self) -> bool:
        '''
        Checks the number of parameters given matches the model function given 
        Can only check for models where not enough parameters are given 
        '''
        assert isinstance(self.dataset['x_data'], ndarray)
        assert isinstance(self.dataset['params'], ndarray)
        try:
            self.model(self.dataset['params'], self.dataset['x_data'])
        except Exception as e:
            if self.printer:
                print(f'ERROR: Model received incorrect number of parameters - {e}')
            return True
        return False
    def _ndof_warning(self) -> None:
        '''
        Warns when the number of degrees of freedom is lower than three 
        when lower than three this signifys a bad fit
        '''
        assert isinstance(self.ndof, int)
        if self.ndof < 3:
            if self.printer:
                print('WARNING: Number of degrees of freedon is less than 3')
    def _initial_checks(self) -> bool:
        '''
        pretty self explanatory compounds all checks into one function so main run function isnt too complicated
        also so any new functions checking things can be added easily without changing the main run() code
        '''
        if self._convert_data_to_arrays():
            return True
        if self._convert_errors_to_arrays():    
            return True
        if self._convert_params_to_array():
            return True
        self._ndof_calculator()
        if self._length_check():
            return True
        if self._parameter_check():
            return True
        if self._error_negativity_check():
            return True
        if self._model_check():
            return True
        self._ndof_warning()
        return False
    def _failed_result(self):
        assert isinstance(self.nparams, int)
        self.success = False
        self.params = np.full(self.nparams, np.nan)
        self.param_errors = np.full(self.nparams, np.nan)
        self.chi2 = np.nan
        ## DO NOT REMOVE return self it WILL BREAK THE CODE
        return self
    def _ols_fit(self) -> bool:
        '''
        Basic OLS fitting function
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
        To be used when x errors are insignificant compared to the y errors
        True means success
        '''
        assert isinstance(self.dataset['params'], ndarray)
        result = least_squares(self.fmin, self.dataset['params'], args=(self.dataset['x_data'],
                                                                        self.dataset['y_data'],
                                                                        self.dataset['x_error'],
                                                                        self.dataset['y_error'],
                                                                        self.model,
                                                                        self.diff))
        self.label = 'OLS'
        if not result.success:
            if self.printer:
                print('Fit failed')
            return False
        self.params = result.x
        chi2 = np.sum(result.fun**2)
        assert isinstance(self.ndof, int)
        reduced_chi2: float = chi2 / self.ndof
        self.chi2 = reduced_chi2
        # converted to array for type annotations use
        try:
            jacobian: ndarray = np.asarray(result.jac)
            covariance: ndarray = np.linalg.inv(jacobian.T @ jacobian)
            self.param_errors = np.sqrt(np.diag(covariance))
        except LinAlgError:
            assert isinstance(self.nparams, int)
            self.param_errors = np.zeros(self.nparams)
            if self.printer:
                print('ERROR: Parameter errors could not be calculated')
        return True
    def _odr_fit(self) -> bool:
        '''
        Base ODR function
        https://hugomvale.github.io/odrpack-python/reference/#odrpack.OdrResult
        True means success
        '''
        self.label = 'ODR'
        # Odr model uses x_data then params whereas least squares uses params then x_data so model is flipped
        def odr_model(x_data: ndarray, params: ndarray) -> ndarray:
            return self.model(params, x_data)
        assert isinstance(self.dataset['x_error'], ndarray)
        assert isinstance(self.dataset['y_error'], ndarray)
        assert isinstance(self.dataset['y_data'], ndarray)
        assert isinstance(self.dataset['x_data'], ndarray)
        assert isinstance(self.dataset['params'], ndarray)
        x_weight: ndarray | None = 1/self.dataset['x_error']**2 if self.raw_dataset['x_error'] is not None else None
        y_weight: ndarray | None = 1/self.dataset['y_error']**2 if self.raw_dataset['y_error'] is not None else None
        sol = odr_fit(odr_model,
                      self.dataset['x_data'],
                      self.dataset['y_data'],
                      self.dataset['params'],
                      weight_x=x_weight,
                      weight_y=y_weight)
        if not sol.success:
            if self.printer:
                print('Fit Failed')
            return False
        self.params = sol.beta
        # odr pack finds the errors in the function so no matrix calculation is needed
        self.param_errors = sol.sd_beta
        chi2: float = sol.sum_square
        assert isinstance(self.ndof, int)
        self.chi2 = chi2 / self.ndof
        return True