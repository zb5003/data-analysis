import scipy as sp
import matplotlib.pyplot as plt
from Assorted_Statistics import *
from data_analysis_classes import *
import numpy.random as rnd
import sympy as sy
import time

x = sy.Symbol('x')
m = sy.Symbol('m')
b = sy.Symbol('b')
a = sy.Symbol('a')
f = m * x**2 / (a + x) + b

model = model_construct(f, [m, b, a], [x])

real_param = [5, 3, 2]
guess = sp.asarray([4.5, 3.5, 1.8])

n = 100
sig_a_ma = 0.1
sig = sp.full(n, sig_a_ma)

indep = sp.linspace(0, 1, n)
real = sp.asarray([model.model_num([real_param, [i]]) for i in indep])
meas = sp.asarray([rnd.normal(model.model_num([real_param, [i]]), sig_a_ma) for i in indep])


# fit = levenberg_marquardt(model, guess, meas, indep, sig)
# fit.lev_mar_run()
# print(fit.current_parameters)
# chisqrd = chi_squared(meas, fit.theory, sig)
# print(chisqrd, Q(n - len(real_param), chisqrd))
t1 = time.time()
mc_bootstrap = bootstrap(indep, meas, sig, model, guess, n_sample_sets=1500)
averages, stds, param_data = mc_bootstrap.save_data()
print("The simulation took", str(round(time.time() - t1, 1)), "seconds")
print(averages, stds)

modeled = sp.zeros(n)
for i in range(n):
    modeled[i] = model.model_num([[averages], [indep[i]]])

plt.plot(indep, modeled, linewidth=2, label="fit")
plt.errorbar(indep, meas, yerr=sig_a_ma, fmt='o', label="measured", marker='.', markersize=10)
plt.title(r"Fit of Measurements to the Model $" + str(sy.latex(f)) + "$")
plt.xlabel("Independent Variable")
plt.ylabel("Dependent Variable")
plt.legend()
plt.show()
