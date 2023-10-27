import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import torch

def Belyeva_function(x,v,t):
    f = 1 / (2 * vm) + (x - v * t) * np.cos(2 * np.pi * v / vm)
    return f


def Belyeva_function_at_point(i,min_i,max_i,iv,vm,x,v,t):
    if i >= min_i and i <= max_i and iv >= min_i and iv <= max_i:
        f = Belyeva_function(x,v,t)
    else:
        f = 0
    return f


def exact_solution_a_la_Belyaeva(xmax, vm, N, t):
# returns f_arr, N
# evaluates exact solution of 1D Vlasov equation
# with 0 electric field as proposed by Dr.Yu.Belyeva(RUDN University)
#
# xmax - maximal value of X coordinate
# vm - mimimal vaule of velocity
# N - size of the resulting solution matrix
# t - moment of time

  dx = xmax / N
  dv = 2 * vm / N
  sol_area_size = N

  half_sol_area_size = sol_area_size * 0.625

  f_arr = np.zeros((2 * N, 2 * N))
  min_i = N - half_sol_area_size
  max_i = N + half_sol_area_size
  for i in range(2 * N):
      for iv in range(2 * N):
          x = i * 0.1
          v = vm + dv * iv
          f_arr[i][iv] = Belyeva_function_at_point(i,min_i,max_i,iv,vm,x,v,t)

  Xs = np.linspace(0, xmax, 2 * N)
  Vs = np.linspace(vm, -vm, 2 * N)

  X, V = np.meshgrid(Xs, Vs)
  fig, ax = plt.subplots(subplot_kw={"projection": "3d"})


  surf = ax.plot_surface(X, V, f_arr, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

  plt.xlabel('X')
  plt.ylabel('V')
  # Customize the z axis.
  #ax.set_zlim(-1.01, 1.01)
 # ax.zaxis.set_major_locator(LinearLocator(10))
  # A StrMethodFormatter is used automatically
  #ax.zaxis.set_major_formatter('{x:.02f}')

  # Add a color bar which maps values to colors.
  fig.colorbar(surf, shrink=0.5, aspect=5)
  plt.title('exact solution for t = '+ str(t))
  plt.show()
  N = 2 * N
  return f_arr

def bella(x,v,t):
    vm = -5
    xmax = 10.0
    dt = 0.01

    f = 1 / (2 * vm)*torch.ones_like(x) + torch.subtract(x,v * t) * torch.cos(2 * np.pi * torch.divide(v,vm))

    return f
    #f_t0 = exact_solution_a_la_Belyaeva(xmax, vm, N, t)


if __name__ == '__main__':
    N = 64

    vm = -5
    xmax = 10.0
    dt = 0.01

    f_t0    = exact_solution_a_la_Belyaeva(xmax,vm,N,0.0)
    f_t0_plus_dt = exact_solution_a_la_Belyaeva(xmax, vm, N, dt)
    dfdt = (f_t0_plus_dt - f_t0)/dt
    #TODO: express time derivative through sympy and plot it
    #express spatial derivative through sympy and plot it
    # make an expression with indefinite coefficients for spatial derivative
    # and minimize difference with exact value through Numpy-based neural network
    # https://kitchingroup.cheme.cmu.edu/blog/2018/11/02/Solving-coupled-ODEs-with-a-neural-network-and-autograd/
    x = torch.linspace(0,xmax,N)
    v = torch.linspace(vm,-vm,N)
    X, V = np.meshgrid(x.numpy(),v.numpy())
    xx,vv = torch.meshgrid(x,v)
    f = bella(xx, vv, 0.0)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    surf = ax.plot_surface(X, V, f.numpy(), cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    qq = 0

    from PDE_loss_function import loss_function

    t = torch.linspace(0,1.0,10)

  #  loss = loss_function(x,v,t,julia)

    # for ti in t:
    #     for xi in x:
    #         for vi in v:
    #             qq = loss_function


