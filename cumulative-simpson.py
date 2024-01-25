import numpy as np
import matplotlib.pyplot as plt

def calc_cumulative_simpson(x_range, f_vals):
    n     = len(x_range)
    start = 0
    stop  = n-2 if n%2==1 else n-3
    de    = (x_range[-1]-x_range[0])/(n-1)
    outs  = np.zeros(n)
    outs[start:stop:2]     += f_vals[start:stop:2]
    outs[start+1:stop+1:2] += 4*f_vals[start+1:stop+1:2]
    outs[start+2:stop+2:2] += f_vals[start+2:stop+2:2]
    outs *= de/3.
    # https://en.wikipedia.org/wiki/Simpson%27s_rule#Composite_Simpson's_rule_for_irregularly_spaced_data
    if n%2==0:
        alpha = 5/12
        beta  = 2/3
        eta   = 1/12
        outs[-1] += alpha*f_vals[-1]*de
        outs[-2] += beta*f_vals[-2]*de
        outs[-3] += -eta*f_vals[-3]*de
    return np.cumsum(outs)

if __name__ == "__main__":
    x_min = -1
    x_max = 1
    x_num = 1001
    x_range = np.linspace(x_min,x_max,x_num)
    f_vals = x_range**2
    integrated_vals = calc_cumulative_simpson(x_range, f_vals)

    fig, ax = plt.subplots(1,1,tight_layout=True)
    ax.set_xlim(x_min, x_max)
    ax.plot(x_range, f_vals, label=r"$f(x)$")
    ax.plot(x_range, integrated_vals, label=r"$\int f(x)dx$")
    ax.legend()
    fig.savefig("example.pdf")