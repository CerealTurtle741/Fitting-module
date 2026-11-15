import sympy as sp
from collections.abc import Callable
from numpy import ndarray
from sympy import parse_expr


class FitModels:
    """
    WARNING sympify uses the eval command 
    Input should not be used unsanitised on web applications
    or apps that allow users to import models from others
    """
    # All models must use x 
    x: sp.Symbol = sp.symbols('x')
    def __init__(
            self,
            model: str
            ) -> None:
        self.str_model: str = model
        
        try:
            self.expr = parse_expr(self.str_model) 
        except SyntaxError:
            raise ModelError('Expression could not be turned into a model')
        
        self.diff_expr = sp.diff(self.expr, self.x) 
        # as free_symbols is a set x must be removed
        self.parameter_symbols: set[sp.Symbol] = self.expr.free_symbols - {self.x}
        
        self.model_func: Callable[[ndarray, ndarray], ndarray] = sp.lambdify(
            [self.parameter_symbols, self.x],
            self.expr,
            'numpy'
            )

        self.diff_func: Callable[[ndarray, ndarray], ndarray] = sp.lambdify(
            [self.parameter_symbols, self.x],
            self.diff_expr,
            'numpy'
            )

class ModelError(Exception):
    """An exception for when a sympy model cannot be created from a string"""
