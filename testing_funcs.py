import numpy as np
from nedler_mead import nelder_mead
from adaptive_gradient_descent import adaptive_coordinate_descent

def f(x):
    return (x[0]-3)**2 + (x[1]+1)**2

def grad(x):
    return np.array([2*(x[0]-3), 2*(x[1]+1)])

def rosenbrock4(x):
    x = np.asarray(x)
    return np.sum(100*(x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
def rosenbrock4_grad(x):
    x = np.asarray(x)
    n = len(x)
    g = np.zeros_like(x)

    for i in range(n):
        if i == 0:
            g[i] = (
                -400 * x[i] * (x[i+1] - x[i]**2)
                - 2 * (1 - x[i])
            )
        elif i == n - 1:
            g[i] = 200 * (x[i] - x[i-1]**2)
        else:
            g[i] = (
                200 * (x[i] - x[i-1]**2)
                -400 * x[i] * (x[i+1] - x[i]**2)
                - 2 * (1 - x[i])
            )

    return g

def rosenbrock_noisy(x, noise_std=0.01):
    true_val = rosenbrock4(x)
    noise = np.random.normal(0, noise_std)
    return true_val + noise

def numerical_grad(x):
    step=1e-6
    g = np.zeros_like(x)
    for i in range(len(x)):
        dx = np.zeros_like(x)
        dx[i] = step
        g[i] = (rosenbrock_noisy(x + dx) - rosenbrock_noisy(x - dx)) / (2*step)
    return g

if __name__ == "__main__":


    x0 = np.array([0.5,0.5,0.5,0.5])
    #solution, hist = adaptive_coordinate_descent(rosenbrock_noisy, numerical_grad, x0)

    best_pt, best_val = nelder_mead(
        lambda x0: f(x0),
        x0=np.array([1.05,0.98,0.95,1]),
        step=0.001,
        max_iter=200,
        tol=1e-5,
        alpha=1.0,
        gamma=2.0,
        rho=0.5,
        sigma=0.5
    )
    print("Best point:", best_pt)
    print("Best value:", best_val)
    # print("Solution:", solution)
    # print("Final objective:", hist[-1])