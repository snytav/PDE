from sympy import *

x,v,t = symbols('x,v,t')
vm = Symbol('vm')

f = 1 / (2 * vm) + (x - v * t) * cos(2 * pi * v / vm)
print(diff(f,t)+v*diff(f,x))