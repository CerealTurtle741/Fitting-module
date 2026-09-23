# Fitting-module
A class made for fitting data

21/09/26:
1st commit:
Class to perform a fit of a set of data
The aim is to use scipys least_squares function to find the parameters, errors and chi2 of a set of data
Least squares can be used for two cases when the errors on x are insignificant (small) compared to y or when no errors are given
When the errors on x are significant an odr fit should be used instead
The fit will be able to have both methods being able to be selected and will have an auto mode where the x errors will be compared to the y errors to determine a better fit 
-- system to be created later 

to start making the class all the variables will be turned into global variables to be used in all functions

Firstly a base straight line function was created and a function to find the residuals 

22/09/26
2nd Commit
Function to convert data to arrays created : True means failed
Data restructured to be inside a dictionary
Length checker added : True means failed
Error converter added 

3rd Commit
Error message added to length check function
Parameter check function
Fixed type annotations
error negativity check added : True means failed
Function to check if the model is passed the right amount of parameters added : True means failed

23/09/26
4th Commit
Full type annotations added some parts fixed with VS code inbuilt Fix function
ndof (number of degrees of freedom) warning added
Initial checks function added to put all checks into one place 
NOTE: When using assert within the class do not use plain assert use assert isinstance(variable, type) to prevent errors from triggering due to truth values

5th Commit
Base ols function added
assert bug fixed in _length_check and _ndof_warning
BUG FIXES:
error array handling put in own function
params taken out of data array function and into its own function
ndof calculator function added

6th Commit
Base odr function added