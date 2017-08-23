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

# real_param = [5, 3]
indep = sp.linspace(0, 10, 50)
real = sp.asarray([5 * i + 3 for i in indep])
meas = sp.asarray([rnd.normal((5 * i + 3), 0.01) for i in indep])
sig = sp.full(50, 0.01)

guess = sp.asarray([4.5, 3.5])
fit = levenberg_marquardt(model, guess, meas, indep, sig)
print(ass.chi_squared(meas, real, sig))
print(fit.cs)
print(fit.cs_error)
fit.lev_mar_run()
print(fit.lam)
print(fit.current_parameters)
# print(model.model, model.model_num([[4.998, 3.013], [0]]))
theory2 = sp.asarray([fit.current_parameters[0] * i + fit.current_parameters[1] for i in indep])

plt.plot(indep, real)
plt.plot(indep, fit.theory)
plt.plot(indep, theory2)
# plt.errorbar(indep, meas, yerr=1)
plt.show()
