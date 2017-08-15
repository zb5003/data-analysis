from ParameterFitting import Levenberg_Marquardt as LM
from ParameterFitting import Monte_Carlo as MC
import scipy as sp
import sympy as sy
import numpy.random as rnd
import matplotlib.pyplot as plt


def mod():
    x = sy.Symbol("x")
    a = sy.Symbol("a")
    b = sy.Symbol("b")
    c = sy.Symbol("c")
    # return c / sy.pi * a / (c**2 + (x - b)**2), x, a, b, c
    return sy.sin(x)**2 / a**2 + sy.cos(x)**2 / b**2 + c, x, a, b, c

def mod_num(px, pa, pb, pc):
    model, x, a, b, c = mod()
    f = sy.lambdify((x, a, b, c), model, "numpy")
    return f(px, pa, pb, pc)

npoints = 100

model, xx, a, b, c = mod()
ex = LM(model, xx, [a, b, c])
parameter = sp.asarray([1.70, 1.8, 0])  # guess
param1 = sp.asarray([1.782, 1.785, 0])  # real values
x = 30 * sp.pi / 180 + sp.asarray(sorted(30 * sp.pi / 180 * rnd.rand(npoints)))  # inputs of the independent variable

sig = sp.full((npoints), 0.0005, dtype=float)
meas = sp.asarray(rnd.normal(mod_num(x, *param1), sig))  # measured value centered on each true value with input uncertainty sig
true = sp.asarray(mod_num(x, *param1))

test = ex.levmar(meas, sig, x, parameter)
cs = ex.chisquared_num(meas, sig, x, test)
print(test)
print(cs)
print(ex.Q(len(x) - len(parameter), cs))
fit = sp.asarray(mod_num(x, *test))

# sim = MC(x, meas, sig, mod, ['a', 'b', 'c'], param)
# print(sim.stat())

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.errorbar(x, meas, yerr=0.02 * sp.pi / 180, label="measured", linewidth=0.1)
ax1.plot(x, true, label="true", linewidth=0.1)
ax1.plot(x, fit, label="fit", linewidth=0.1)
ax1.legend()
plt.show()
# plt.savefig("Fuck.pdf")
# plt.close()
