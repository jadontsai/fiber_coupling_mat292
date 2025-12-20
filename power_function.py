import numpy as np
import math
import matplotlib.pyplot as plt

from scipy.integrate import nquad
from nedler_mead import nelder_mead

def power_function(x_1, y_1, x_2, y_2, x_i, y_i, w, A, c, k):
    """
    Computes the optical power via a 2D complex integral.

    Inputs (floats unless otherwise):
    x_1, y_1: beam 1 angles
    x_2, y_2: beam 2 angles
    x_i, y_i: incident angles
    w: Beam waist
    A: Amplitude scaling
    c: geometric constant
    k:Wave number

    Outputs:
    power
    """
    a = c*(2*x_1 - x_i)
    b = c*(2*y_1 - y_i)

    theta_x = 2*x_1 + x_2 - x_i - math.pi/2
    theta_y = 2*y_1 + y_2 - y_i - math.pi/2

    w_sq = pow(w, 2)

    exponential = -pow(a/w, 2) - pow(b/w, 2) + w_sq/8*pow(2*a/w_sq - 1j*k*theta_x, 2) + w_sq/8*pow(2*b/w_sq - 1j*k*theta_y, 2)    
    val = pow(abs(A*math.pi*w_sq/2*math.exp(exponential)), 2) #debugging
    print(f"value={val}")#debugging
    return val

power_history = [] #for plotting

def objective(x):
    print("x has dimension ", x.shape) #for debugging
    x1, y1, x2, y2 = x
    pi_2 = math.pi /2
    power = power_function(
        x1, y1, x2, y2,
        x_i=pi_2,
        y_i=pi_2,
        w=1,
        A=1.0,
        c=0.1,
        k=1
    )
    power_history.append(power)
    return -power  

if __name__ == '__main__':

    pi_2 = math.pi /2
    pi_4 = math.pi/4
    x0 = np.array([pi_4,pi_4, pi_2, pi_2])
    best_pt, best_val = nelder_mead(
        objective,
        x0=x0,
        step=0.01,
        max_iter=200,
        tol=1e-10,
        alpha=2,
        gamma=2,
        rho=0.5,
        sigma=0.5
    )

    print("Best point:", best_pt)
    print("Max power:", -best_val)
    plt.figure()
    plt.plot(power_history)
    plt.xlabel("Iteration number")
    plt.ylabel("Power")
    plt.title(f"Power vs iterations for {x0}")
    plt.grid(True)
    plt.show()
