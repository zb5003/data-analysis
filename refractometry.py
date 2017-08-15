from ParameterFitting import Levenberg_Marquardt as LM
from ParameterFitting import Monte_Carlo as MC
import scipy as sp
import sympy as sy
import numpy.random as rnd
import matplotlib.pyplot as plt

def mod():
    x = sy.Symbol("x")
    nx = sy.Symbol("nx")
    nz = sy.Symbol("nz")
    # return sy.cos(x)**2 / nz**2 + sy.sin(x)**2 / nx**2, x, nx, nz
    return (1 / nz**2 - 1 / nx**2) * sy.cos(x)**2 + (1 / nx**2), x, nx, nz

def mod_num(px, pa, pb):
    model, x, a, b = mod()
    f = sy.lambdify((x, a, b), model, "numpy")
    return f(px, pa, pb)

npoints = 150
model, xx, a, b = mod()
ex = LM(model, xx, ['nx', 'nz'])
guess = sp.asarray([1.70, 1.80])
real = sp.asarray([1.782, 1.785])
x = sp.asarray(sorted(30 * rnd.rand(npoints))) * sp.pi / 180

sig_val = 0.0003
sig = sp.full((npoints), sig_val, dtype=float)
meas = sp.asarray(rnd.normal(mod_num(x, *real), sig))  # measured value centered on each true value with input uncertainty sig
true = sp.asarray(mod_num(x, *real))

test = ex.levmar(meas, sig, x, guess)
cs = ex.chisquared_num(meas, sig, x, test)
fit = sp.asarray(mod_num(x, *test))
print(test)
print(cs)
print(ex.Q(len(x) - len(guess), cs))

sim = MC(x, xx, meas, sig, model, ['nx', 'nz'], guess)
one, two, three, four, five = sim.stat()
print(two, three, four, five)

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.errorbar(x, meas, yerr=sig_val * sp.pi / 180, label="measured", linewidth=0.5)
ax1.plot(x, true, label="true", linewidth=0.5)
ax1.plot(x, fit, label="fit", linewidth=0.5)
ax1.legend()
plt.show()
# plt.savefig("Fuck.pdf")
# plt.close()
