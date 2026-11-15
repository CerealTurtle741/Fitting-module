import numpy as np
from scipy.optimize import least_squares 
from odrpack import odr_fit
from time import time
from collections.abc import Callable
from numpy.typing import ArrayLike
from numpy import ndarray
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt
from typing import Self
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from warnings import warn
import fitmodels
from dataclasses import dataclass


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
    Calculating the residuals of the data 
    (vertical distance between the data point and its predicted value)
    Used for least squares fitting 
    '''
    y_model: ndarray = model(params, x_data)
    y_model_diff: float | ndarray = diff(params, x_data)
    # x error is included by weighting it with the derivative of the model
    y_diff_total: ndarray = np.sqrt(y_error**2 + x_error**2 * y_model_diff**2)
    # if y_diff_total is 0 replace it with one to stop divide by zero errors
    y_diff_total = np.where(y_diff_total == 0, 1, y_diff_total)
    return (y_data - y_model) / y_diff_total


@dataclass
class Data:
    x_data: ndarray
    y_data: ndarray
    initial_params: ndarray
    x_error: ndarray
    y_error: ndarray

    @classmethod
    def from_raw(
        cls,
        x_data: ArrayLike,
        y_data: ArrayLike,
        initial_params: ArrayLike,
        x_error: None | ArrayLike = None,
        y_error: None | ArrayLike = None
        ) -> Self:
        x = to_array(x_data, 'x_data')
        y = to_array(y_data, 'y_data')
        p = to_array(initial_params, 'initial_params')
        dx = to_error(x_error, x, 'x_error')
        dy = to_error(y_error, x, 'y_error')
        data = cls(x, y, p, dx, dy)
        data.check()
        return data

    def _length_check(self) -> None:
        """
        Check the data lengths are all the same 
        """
        try:
            x_data_length: int = len(self.x_data)
            y_data_length: int = len(self.y_data)
            x_error_length: int = len(self.x_error)
            y_error_length: int = len(self.y_error)
        except TypeError:
            raise TypeError(
                'Length of dataset could not be calculated\n'
                'CHECK _length_check code\n'
                'This error should not be achievable'
                )
        
        if not (x_data_length == y_data_length == 
                x_error_length == y_error_length):
            raise FitError(
                f'Data arrays supplied are not the same length\n'
                f'x: {x_data_length} y: {y_data_length} '
                f'x error: {x_error_length} y error: {y_error_length}'
                )

    def _parameter_check(self) -> None:
        """
        making sure there is enough data points for the number of parameters
        e.g. 3 for a linear relationship
        """
        if len(self.x_data) < self.nparams + 1:
            raise FitError(
                'Number of data points is less the number of parameters'
                )

    def _error_negativity_check(self) -> None:
        '''
        the errors cannot be negative 
        otherwise it will break the fitting models
        '''
        for name, error in [('x_error', self.x_error), ('y_error', self.y_error)]:
            if np.any(error < 0):
                raise FitError(
                    f'{name} cannot contain negative numbers'
                    )

    def _ndof_warning(self) -> None:
        '''
        Warns when the number of degrees of freedom is lower than three
        when lower than three this signifys a bad fit
        '''
        if self.ndof < 3:
            warn(
                'WARNING: Number of degrees of freedon is less than 3',
                stacklevel=2
                )

    def check(self):
        self._length_check()
        self._parameter_check()
        self._error_negativity_check()
        self._ndof_warning()

    @property
    def ndof(self) -> int:
        return len(self.x_data) - len(self.initial_params)

    @property
    def nparams(self) -> int:
        return len(self.initial_params)


def to_array(array: ArrayLike, name: str) -> ndarray:
    try:
        return np.asarray(
            array,
            dtype=float)
    except ValueError:
        raise ValueError(f'{name} could not be converted to an array')
    except TypeError:
        raise TypeError(f'{name} could not be converted to a float')


def to_error(array: ArrayLike | None, x: ndarray, name: str) -> ndarray:
    if array is None:
        return np.zeros_like(x)
    if np.isscalar(array):
        return np.full_like(x, array, dtype=float)
    return to_array(array, name)


class Fit:
    def __init__(
            self,
            x_data: ArrayLike,
            y_data: ArrayLike,
            initial_params: ArrayLike,
            x_error: None | ArrayLike = None,
            y_error: None | ArrayLike = None,
            model: Callable[[ArrayLike, ArrayLike], ndarray] | str = straight_line_model,
            diff: Callable[[ArrayLike, ArrayLike], ndarray | float] | None = straight_line_diff,
            fmin: Callable[..., ndarray] = minimise,
            printer: bool=True
            ) -> None:

        self.dataset: Data = Data.from_raw(x_data, y_data, initial_params, x_error, y_error)
        self.fmin: Callable[..., ndarray] = fmin
        self.printer: bool = printer

        # Models
        if callable(model) and callable(diff):
            self.model: Callable[[ndarray, ndarray], ndarray] = model
            self.diff: Callable[[ndarray, ndarray], float | ndarray] = diff
        elif isinstance(model, str):
            model_result = fitmodels.FitModels(model)
            self.model = model_result.model_func
            self.diff = model_result.diff_func
        elif callable(model) and diff is None:
            raise fitmodels.ModelError('diff cannot be None if model is a function')

        # Create Values to be updated after checks
        self.success: None | bool = None
        self._chi2: None | float = None
        self._params: None | ndarray = None
        self._param_errors: None | ndarray = None
        self.label: None | str = None

        # Values for plotting
        self.plotted: bool = False
        self.fig: None | Figure = None
        self.ax: None | Axes = None
    
    # Run function
    def run(self, method: str = 'ols') -> Self:
        self.method: str = method
        start_time: float = time()
        self._model_check()
        
        method_list: dict[str, Callable[[], None]] = {
            'ols':self._ols_fit,
            'odr':self._odr_fit
            }
        
        if method.lower() not in method_list:
            raise FitError(
                'Method not found\n'
                f'Valid method are {method_list.keys()}'
                )
        
        method_list[method.lower()]()
        end_time: float = time()
        time_taken: float = end_time - start_time
        if self.printer:
            print(f'Method: {self.label}')
            print(f'Time taken: {time_taken}')

        if self.printer:
            print(f'Params: {self._params}')
            print(f'Param errrors: {self._param_errors}')
            print(f'Reduced Chi2: {self._chi2}')
        ## DO NOT REMOVE return self it WILL BREAK THE CODE
        return self

    def _model_check(self) -> None:
        '''
        Checks the number of parameters given matches the model function given
        Can only check for models where not enough parameters are given
        '''
        try:
            self.model(self.dataset.initial_params, self.dataset.x_data)
        except TypeError:
            raise TypeError(
                'Model must be written in form:\n'
                'model(params, x_data)\n'
                '(Where params is an arraylike structure of the parameters)'
                )
        except ValueError:
            raise ValueError(
                'Not enough parameters given to unpack array'
                )
        except IndexError:
            raise IndexError(
                'Model indexes parameter outside length of array'
                )
    
    def _ols_fit(self) -> None:
        '''
        Basic OLS fitting function
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
        To be used when x errors are insignificant compared to the y errors
        True means success
        '''

        result = least_squares(
            self.fmin,
            self.dataset.initial_params,
            args=(
                self.dataset.x_data,
                self.dataset.y_data,
                self.dataset.x_error,
                self.dataset.y_error,
                self.model,
                self.diff
                ))
        
        self.label = 'OLS'
        self.success = result.success
        if not result.success:
            raise FitError(f'Fit Failed - {result.message}')
        
        
        self._params = result.x
        chi2 = np.sum(result.fun**2)
        reduced_chi2: float = chi2 / self.dataset.ndof
        self._chi2 = reduced_chi2

        # converted to array for type annotations use
        try:
            jacobian: ndarray = np.asarray(result.jac)
            covariance: ndarray = np.linalg.inv(jacobian.T @ jacobian)
            self._param_errors = np.sqrt(np.diag(covariance))
        except LinAlgError:
            self._param_errors = np.zeros(self.dataset.nparams)
            warn('Parameter errors could not be calculated', stacklevel=2)

    def _odr_fit(self) -> None:
        '''
        Base ODR function
        https://hugomvale.github.io/odrpack-python/reference/#odrpack.OdrResult
        True means success
        '''
        self.label = 'ODR'
        # Odr model flipped from inputted function signature
        def odr_model(x_data: ndarray, params: ndarray) -> ndarray:
            return self.model(params, x_data)
        
        x_error = self.dataset.x_error
        y_error = self.dataset.y_error
        x_weight: ndarray | None = None
        y_weight: ndarray | None = None
        if not np.any(x_error):
            warn('OLS model should be used as x values are exact', stacklevel=2)
        else:
            smallest_x_error: float = min([error for error in x_error if not error == 0])
            # Replaces any zero values with the smallest error
            # Uses smallest error to keep the point relevent to the fit
            x_error = np.where(x_error==0, smallest_x_error, x_error)
            x_weight = 1/x_error**2

        if not np.any(y_error):
            warn('No Y errors provided', stacklevel=2)
        else:
            if 0 in y_error:
                raise FitError(
                    'y_error cannot contain zero values\n'
                    'Either fix this or use OLS'
                    )
            y_weight = 1/y_error**2

        sol = odr_fit(
            odr_model,
            self.dataset.x_data,
            self.dataset.y_data,
            self.dataset.initial_params,
            weight_x=x_weight,
            weight_y=y_weight
            )
        self.success = sol.success
        if not sol.success:
            raise FitError(f'Fit failed - {sol.stopreason}')
        self.success = True
        self._params = sol.beta
        # odr pack finds the errors in the function 
        # so no matrix calculation is needed
        self._param_errors = sol.sd_beta
        chi2: float = sol.sum_square
        self._chi2 = chi2 / self.dataset.ndof

    # Plot function
    def plot(self) -> tuple[Figure, Axes]:
        '''
        Plot function for if the data has been successfully fitted 
        '''
        if self.printer:
            print('Plot successfully produced')
        fig, ax = plt.subplots(figsize=(8,6))
        x_data = self.dataset.x_data
        y_data = self.dataset.y_data
        x_error = self.dataset.x_error
        y_error = self.dataset.y_error
        
        fmt = '.' if np.any(x_error) or np.any(y_error) else 'o'
        
        ax.errorbar(
            x_data,
            y_data,
            xerr=x_error,
            yerr=y_error,
            fmt=fmt,
            color='black',
            label='Data',
            zorder=3
            )
        
        ax.grid(linestyle='--', zorder=1)
        ax.legend()
        self.fig, self.ax = fig, ax
        self.plotted = True
        # Remember to actually show the plot 
        # plt.show() 
        # dont call plt.show() in the function
        # it prevents any changes from being made outside the function
        # but you MUST remember to call it outside the function
        return fig, ax

    def fit(self) -> tuple[Figure, Axes]:
        fig, ax  = self.fig, self.ax
        if not self.plotted:
            fig, ax = self.plot()
        if self.success is None:
            raise FitError('Data has not been fitted')
        if self.success is False:
            raise PlotError('Fit has failed')
        
        x_data = self.dataset.x_data

        x_model: ndarray = np.linspace(float(min(x_data)), float(max(x_data)), 200) 
        y_model: ndarray = self.model(self.params, x_model) 

        assert (isinstance(ax, Axes) and isinstance(fig, Figure))
        ax.plot(
            x_model,
            y_model,
            marker='',
            linestyle='-',
            label=f'Fit - {self.method.upper()}',
            zorder=2
            )
        ax.legend()
        return fig, ax

    # Save function
    def save(self, file_name: str, file_type: str = 'png') -> None:
        '''
        save function for saving the figure produced by plot()
        the file name must be inputted 
        the file type can be changed but defaults to png 
        '''
        if self.fig is None:
            raise PlotError('Figure must be plotted to save')
        file: str = f'{file_name}.{file_type}'
        try:
            self.fig.savefig(file)
            if self.printer:
                print(f'File saved as: {file}')
        except ValueError:
            raise SaveError('Unsupported file format')
        except PermissionError:
            raise SaveError('No permission to save file')
        except FileNotFoundError:
            raise SaveError('Directory not found')
        except OSError:
            raise SaveError('Invalid file name')
        except TypeError:
            raise SaveError('Invalid argument passed to savefig')
        except Exception as e:
            raise SaveError(f'{e}')

    # property helpers for type annotations
    @property
    def params(self) -> ndarray:
        if self._params is None:
            raise RuntimeError('Fit must be completed before accessing Paramters')
        return self._params
    
    @property
    def param_errors(self) -> ndarray:
        if self._param_errors is None:
            raise RuntimeError('Fit must be completed before accessing Parameter Errors')
        return self._param_errors
    
    @property
    def chi2(self) -> float:
        if self._chi2 is None:
            raise RuntimeError('Fit must be completed before accessing Chi2')
        return self._chi2

    def __repr__(self) -> str:
        args: list[str] = [
            f'x_data={self.dataset.x_data!r}',
            f'y_data={self.dataset.y_data!r}',
            f'initial_params={self.dataset.initial_params!r}']
        # Only add non default values to the repr
        if np.any(self.dataset.x_error):
            args.append(f'x_error={self.dataset.x_error!r}')
        if np.any(self.dataset.y_error):
            args.append(f'y_error={self.dataset.y_error!r}')
        if self.model is not straight_line_model:
            if isinstance(self.model, str):
                args.append(f'model={self.model}')
            if callable(self.model):
                args.append(f'model={self.model.__name__}')
        if self.diff is not straight_line_diff and callable(self.diff):
            args.append(f'diff={self.diff.__name__}')
        if self.fmin is not minimise:
            args.append(f'fmin={self.fmin.__name__}')
        if self.printer is not True:
            args.append(f'printer={self.printer}')
        return f'{type(self).__name__}({', '.join(args)})'


class FitError(Exception):
    """An exception for when the fit fails"""


class PlotError(Exception):
    """An exception for when plotting fails"""


class SaveError(Exception):
    """An exception for when saving fails"""