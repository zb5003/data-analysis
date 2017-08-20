import scipy as sp
from Assorted_Statistics import *
from data_analysis_classes import *
import numpy.random as rnd
import sympy as sy

x = sy.Symbol('x')
y = sy.Symbol('y')
z = sy.Symbol('z')
f = y * x**1 + sy.sin(y) * z**3

model = model_construct(f, [y, z], [x])
# print(sy.diff(f, y, z))
print(model.model, model.model_grad)
print(model.model_num([[2, 2], [0]]))
print(model.model_grad_num([[2, 2], [0]]))
print(model.model_hessian)