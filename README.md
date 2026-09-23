# Fitting Module

A fitting module for finding the parameters from a set of data and its errors. It has two modes that can be selected either doing an Ordinary Least Squares fit (https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html) or an Orthogonal Distance Regression (https://hugomvale.github.io/odrpack-python/). 

## Functions
- run - the method can be selected by inputting either 'ols' or 'odr' into it [defaults to 'ols']
- plot - returns the figure and axes of the plot and remember to call plt.show() when done to see the plot
- save - saves the figure created (a file name must be inputted) must be called before plt.show()

## How to use
1. Assign the Fit class to a variable
2. In the Fit class add the x and y data and the initial parameter guess
3. *[Optional]*: add the x and y errors
4. A straight line model is built in so if a custom model is required the model and its derivative must be defined first
5. The model defaults to printing any error messages and a couple other infomation to the terminal to turn this off set printer to False
6. On the variable that the Fit class was assigned to call the run function (e.g. fit.run() if the odr method is required input it into the brackets of the run function)
7. For plotting and saving figures call the functions in the same way that the run function was called (for the save function remember to add the filename as a string into the save function)
8. After plotting and/or saving call plt.show() to see the figure

## API Reference
```
Fit(
 x_data: ArrayLike,
 y_data: ArrayLike,
 initial_params: ArrayLike,
 x_error: None | ArrayLike = None,
 y_error: None | ArrayLike = None,
 model: Callable[[ArrayLike, ArrayLike], ndarray] = straight_line_model,
 diff: Callable[[ArrayLike, ArrayLike], ndarray | float] = straight_line_diff,
 fmin: Callable[..., ndarray] = minimise,
 printer: bool=True
 )
```
| &nbsp;&nbsp;Parameter&nbsp;&nbsp;| Description <img width="775" height="1"> |
|---|---|
| x_data | X data inputted in any array like data type |
| y_data | Y data inputted in any array like data type |
| initial_params | The guess of the inital parameters for the default model must be inputted in form (intercept, gradient) |
| x_error | X errors inputted in any array like data type [defaults to None] |
| y_error | Y errors inputted in any array like data type [defaults to None] |
| model | The model used for fitting must be inputted in the form model(parameters, x_data) [defaults to a linear model] |
| diff | The derivative of the model [defaults to a linear model] |
| fmin | A function to find the residuals of the data [default function integrated]
| printer | Whether the function can print anything to the terminal |

| &nbsp;&nbsp;&nbsp;Attribute&nbsp;&nbsp;&nbsp;&nbsp; | Description <img width="775" height="1"> |
|---|---|
| params | The parameters calculated from the fit.<br> In the default settings it is in the form [intercept, gradient] |
| param_errors | The errors associated with its parameter |
| chi2 | The reduced chi squared of the fit |
| fig | If plot has been called is the Figure of the successful plot |
| ax | If plot has been called is the Axes of the successful plot |
| dataset | A dictionary that stores all the data and params used by the function<br> Keys for the dictionary: ['x_data', 'y_data', 'x_error', 'y_error', 'params'] |
| success | Whether the fit was successful |
