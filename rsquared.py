import pandas as pd
df = pd.read_csv("vulcancoin.csv", delimiter=",")


from sklearn.linear_model import LinearRegression
model = LinearRegression()
trans_amt=row[3].split()[0]
trans_fee=row[4]
X, y = trans_amt,trans_fee
model.fit(X, y)

# compute with formulas from the theory
yhat = model.predict(X)
SS_Residual = sum((y-yhat)**2)       
SS_Total = sum((y-np.mean(y))**2)     
r_squared = 1 - (float(SS_Residual))/SS_Total
adjusted_r_squared = 1 - (1-r_squared)*(len(y)-1)/(len(y)-X.shape[1]-1)
print(r_squared, adjusted_r_squared)
# 0.877643371323 0.863248473832

# compute with sklearn linear_model, although could not find any function to compute adjusted-r-square directly from documentation
print(model.score(X, y), 1 - (1-model.score(X, y))*(len(y)-1)/(len(y)-X.shape[1]-1))
# 0.877643371323 0.863248473832 


### different ways

import numpy

# Polynomial Regression
def polyfit(x, y, degree):
    results = {}

    coeffs = numpy.polyfit(x, y, degree)

     # Polynomial Coefficients
    results['polynomial'] = coeffs.tolist()

    # r-squared
    p = numpy.poly1d(coeffs)
    # fit values, and mean
    yhat = p(x)                         # or [p(z) for z in x]
    ybar = numpy.sum(y)/len(y)          # or sum(y)/len(y)
    ssreg = numpy.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = numpy.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    results['determination'] = ssreg / sstot

    return results

### heres one:
def rsquared(x, y):
    """ Return R^2 where x and y are array-like."""

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    return r_value**2


### or this:

from sklearn.metrics import r2_score

coefficient_of_dermination = r2_score(y, p(x))

### one liner if it comes to this:

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)