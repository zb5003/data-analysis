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

real_param = [5, 3]
indep = sp.linspace(0, 10, 50)
meas = sp.asarray([model.model_num([[5, 3], i]) for i in indep])
sig = sp.full(50, 0.1)

guess = sp.asarray([4.9, 3.02])
fit = levenberg_marquardt(model, guess, meas, sig)
print(fit.cs_error)
fit.lev_mar_run()
print(fit.lam)
print(fit.current_parameters)

# plt.plot(indep, meas)
# plt.show()
