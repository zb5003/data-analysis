import scipy as sp
import matplotlib.pyplot as plt
from Assorted_Statistics import *
from data_analysis_classes import *
import numpy.random as rnd
import sympy as sy

x = sy.Symbol('x')
m = sy.Symbol('m')
b = sy.Symbol('b')
a = sy.Symbol('a')
f = m * x**2 / ( a + x) + b

model = model_construct(f, [m, b, a], [x])

real_param = [5, 3, 0.03]
guess = sp.asarray([4.5, 3.5, 1.99])

n = 100
sig_a_ma = 0.1
sig = sp.full(n, sig_a_ma)

indep = sp.linspace(0, 1, n)
real = sp.asarray([model.model_num([real_param, [i]]) for i in indep])
meas = sp.asarray([rnd.normal(model.model_num([real_param, [i]]), sig_a_ma) for i in indep])


fit = levenberg_marquardt(model, guess, meas, indep, sig)
fit.lev_mar_run()
print(fit.current_parameters)

plt.plot(indep, real, label="real")
plt.plot(indep, fit.theory, label="fit")
# plt.plot(indep, theory2)
plt.errorbar(indep, meas, yerr=sig_a_ma, label="measured")
plt.legend()
plt.show()
