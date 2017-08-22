import scipy as sp
import matplotlib.pyplot as plt
from Assorted_Statistics import *
from data_analysis_classes import *
import numpy.random as rnd
import sympy as sy

x = sy.Symbol('x')
m = sy.Symbol('m')
b = sy.Symbol('b')
f = m * x + b

model = model_construct(f, [m, b], [x])
print(model.model_hessian)

# real_param = [5, 3]
indep = sp.linspace(0, 10, 50)
real = sp.asarray([5 * i + 3 for i in indep])
meas = sp.asarray([rnd.normal((5 * i + 3), 1) for i in indep])
sig = sp.full(50, 1)

guess = sp.asarray([4.9, 2.9])
fit = levenberg_marquardt(model, guess, meas, sig)
print(fit.cs_error)
fit.lev_mar_run()
print(fit.lam)
print(fit.current_parameters)

# plt.plot(indep, real)
# plt.errorbar(indep, meas, yerr=1)
# plt.show()
