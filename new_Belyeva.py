import autograd.numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import torch

vm = -5
xmax = 4
dt = 0.01

def Belyeva_function(x,v,t):
    f = 1 / (2 * vm) + (x - v * t) * np.cos(2 * np.pi * v / vm)
    return f




if __name__ == '__main__':
    N = 8
    xmax = 4.0
    x = torch.linspace(0, xmax, N)
    v = torch.linspace(vm, -vm, N)
    X, V = np.meshgrid(x.numpy(), v.numpy())
    xx, vv = torch.meshgrid(x, v)
    f = np.zeros((N,N))
    t = 0.0
    for i,xi in enumerate(x):
        for j,vi in enumerate(v):
            f[i][j] = Belyeva_function(xi,vi,i)
    #f = bella(xx, vv, 0.0)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    surf = ax.plot_surface(X, V, f, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    plt.xlabel('X')
    plt.ylabel('V')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    qq = 0
