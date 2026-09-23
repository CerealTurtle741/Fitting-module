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

7th Commit
assert declarations tidied up to only take one line in their respective functions
run function created
INTERESTING NOTE:
- when timed odr_fit was 3-4 times quicker even when its task was changed to 'OLS' but needs further testing as was only tested with basic data (seen below)
x=[0.9,1.8,3.4,4.1,4.9]
y=[2.3,4.1,5.7,8.2,9.5]
dx=0.2
dy=0.1
p=[0,2]
fit=Fit(x,y,p,dx,dy)
fit.run('odr')
print('Parameters: ',fit.params)
print('Chi2: ',fit.chi2)
Method: OLS
Time taken: 0.001956939697265625
Parameters:  [0.49488934 1.80963929]
Chi2:  2.7204972293115457
Method: ODR
Time taken: 0.00064849853515625
Parameters:  [0.49488832 1.80963961]
Chi2:  2.7204972293301797
Method: ODR - task='OLS'
Time taken: 0.0005497932434082031
Parameters:  [0.65966009 1.75507942]
Chi2:  37.2823543898539
- However the Chi2 is massive so greater testing is required to see if odr_pack is more viable speed wise
- A great amount of logic would probably be required to see which method is better when an auto model is built

8th Commit
Bug in run fixed - return _failed_result (returns the function) -> return _failed_result()
Added plot function
- lots of # ignore added as the problems are purely pylance not allowing unknown types