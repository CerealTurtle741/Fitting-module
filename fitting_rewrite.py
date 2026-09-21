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

class Fit:
    '''
    21/09/26:
    Class to perform a fit of a set of data
    The aim is to use scipys least_squares function to find the parameters, errors and chi2 of a set of data
    Least squares can be used for two cases when the errors on x are insignificant (small) compared to y or when no errors are given
    When the errors on x are significant an odr fit should be used instead
    The fit will be able to have both methods being able to be selected and will have an auto mode where the x errors will be compared to the y errors to determine a better fit 
    -- system to be created later 
    '''
    def __init__(self) -> None:
        pass