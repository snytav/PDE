# -*- coding: utf-8 -*-
"""PDE example (4).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1bhu2M0prME2VtUjbDiaM2pcXmQmSsPyz
"""

# Commented out IPython magic to ensure Python compatibility.
import autograd.numpy as np
import torch
from autograd import grad, jacobian
import autograd.numpy.random as npr

from matplotlib import pyplot as plt
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D
# %matplotlib inline
from exact_solution import bella

vm = -5
xmax = 4
nx = 8
ny = nx

c = 1.0

dx = 1. / nx
dy = 1. / ny

x_space = np.linspace(0, xmax, nx)
y_space = np.linspace(0, 1.0, ny)


from new_Belyeva import Belyeva_function
def f(x):
    y = Belyeva_function(x[0],c,x[1])
    return y

def analytic_solution(x):
     return f(x)

def psy_analytic(x,net_out):
    return f(x)
#    return (1 / (np.exp(np.pi) - np.exp(-np.pi))) * \
#    		np.sin(np.pi * x[0]) * (np.exp(np.pi * x[1]) - np.exp(-np.pi * x[1]))
surface = np.zeros((ny, nx))

for i, x in enumerate(x_space):
    for j, y in enumerate(y_space):
        surface[i][j] = analytic_solution([x, y])

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
X, Y = np.meshgrid(x_space, y_space)
surf = ax.plot_surface(X, Y, surface, rstride=1, cstride=1, cmap=cm.viridis,
        linewidth=0, antialiased=False)

# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)
# ax.set_zlim(0, 2)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$');



def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def neural_network(W, x):
    a1 = sigmoid(np.dot(x, W[0]))
    return np.dot(a1, W[1])


def neural_network_x(x):
    a1 = sigmoid(np.dot(x, W[0]))
    return np.dot(a1, W[1])

def A(x):
    return f(x)


def psy_trial(x, net_out):
    return A(x) + x[0] * (xmax - x[0]) * x[1] * (1 - x[1]) * net_out


def loss_function(W, x, y,psy):
    loss_sum = 0.

    for xi in x:
        for yi in y:

            input_point = np.array([xi, yi])

            net_out = neural_network(W, input_point)[0]

            net_out_jacobian = jacobian(neural_network_x)(input_point)
            net_out_hessian = jacobian(jacobian(neural_network_x))(input_point)

            psy_t = psy(input_point, net_out)
            psy_t_jacobian = jacobian(psy)(input_point, net_out)
            psy_t_hessian = jacobian(jacobian(psy))(input_point, net_out)

            gradient_of_trial_dx = psy_t_jacobian[0]
            gradient_of_trial_dy = psy_t_jacobian[1]
            gradient_of_trial_d2x = psy_t_hessian[0][0]
            gradient_of_trial_d2y = psy_t_hessian[1][1]

            # func = f(input_point) # right part function

            err_sqr = np.abs(gradient_of_trial_dx +  gradient_of_trial_dy )
            if err_sqr > loss_sum:
                loss_sum = err_sqr

    return loss_sum



W = [npr.randn(2, nx), npr.randn(nx, 1)]

ideal_loss = loss_function(W, x_space, y_space,psy_analytic)

lmb = 1e-2  ## USUAL VALUE RESULTS IN INSTABILITY

print(neural_network(W, np.array([1, 1])))

loss = loss_function(W, x_space, y_space,psy_trial)
i = 0
while loss > 1e-0:
    loss_grad =  grad(loss_function)(W, x_space, y_space,psy_trial)
    loss = loss_function(W, x_space, y_space,psy_trial)

    W[0] = W[0] - lmb * loss_grad[0]
    W[1] = W[1] - lmb * loss_grad[1]
    print(i,loss)
    i = i + 1



print(loss_function(W, x_space, y_space,psy_trial))

surface2 = np.zeros((ny, nx))
surface = np.zeros((ny, nx))

for i, x in enumerate(x_space):
 for j, y in enumerate(y_space):
     surface[i][j] = analytic_solution([x, y])

for i, x in enumerate(x_space):
 for j, y in enumerate(y_space):
     net_outt = neural_network(W, [x, y])[0]
     surface2[i][j] = psy_trial([x, y], net_outt)


print(surface[2])
print(surface2[2])
diff = np.max(np.abs(surface-surface2))
print('difference between analytic and NN ',diff)


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
X, Y = np.meshgrid(x_space, y_space)
surf = ax.plot_surface(X, Y, surface, rstride=1, cstride=1, cmap=cm.viridis,
       linewidth=0, antialiased=False)

# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)
# ax.set_zlim(0, 3)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$');
plt.title('Analytic solution')
plt.show()


fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(x_space, y_space)
surf = ax.plot_surface(X, Y, surface2, rstride=1, cstride=1, cmap=cm.viridis,
       linewidth=0, antialiased=False)

# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)
# ax.set_zlim(0, 3)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$');
plt.title('NN solution')
plt.show()

print(loss_function(W, x_space, y_space,psy_trial))

