import random
import numpy as np
import numpy.linalg as la
print("imported numpy")
import scipy.optimize as so
print("imported scipy.optimize")
print("")

## Parameters used to generate data

a_real = 0.4
b_min = 0.4
b_max = 0.8
b_real = b_min + (b_max - b_min) * random.random()
c_real = 1024

def generate_data(x, a, b, c):
    """
    Generates x,y measurements over the model
    y = a * x**2 + x**b + c
    introducing an error by sampling a normal distribution.
    """
    return (a*np.power(x, 2) + np.power(x, b) + c
            + np.random.normal(size=x.size, scale=100))

x = np.array(range(1000, 450000, 100))
y = generate_data(x, a_real, b_real, c_real)
print("Generated data")

## Functions used to rediscover parameters a, b, c

def fit_b(x, y, b):
    """
    Given a value for b, find the values of a and c
    that minimize the error between measurements x,y
    and the model y = a * x**2 + x**b + c.
    
    Returns the result of numpy.linalg.lstsq.
    """
    A = np.vstack([np.power(x, 2), np.ones(len(x))]).T
    c = y - np.power(x, b)
    solution, residuals, rank, s = la.lstsq(A, c)
    return solution, residuals, rank, s

def fit_b_error(*args):
    """
    Returns just the residue from the least squares method
    used by fit_b.
    """
    x, residuals, *_ = fit_b(*args)
    return residuals[0]

def fit(x, y):
    """
    Computes a fit of a, b, c using function minimization over fit_b.
    """
    min_result = so.minimize_scalar(lambda b: fit_b_error(x, y, b),
            bounds=(b_min, b_max),
            method='Bounded')
    b = min_result.x
    solution, *_ = fit_b(x, y, b)

    a, c = solution

    return a, b, c

a, b, c = fit(x, y)
print("Computed fit")
print("")

## Print values for a, b, c along with the absolute error

print("Solution:")
for k, v, r in (('a', a, a_real), ('b', b, b_real), ('c', c, c_real)):
    print("  %s = %g (error %g)" % (k, v, r - v))
