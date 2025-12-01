import numpy as np

def adaptive_coordinate_descent(f, grad, x0, step0=0.1,increase_factor=1.05,decrease_factor=0.5,max_iters=5000,tol=1e-6):
    """
    Adaptive Coordinate Descent for finding rough area of maximum power (nedler mead used after)
    Parameters:
    f: objective function (testing on rosenbruck function)
    grad: Gradient of objective
    x0: init guess (np array)
    step0: initial step size
    increase_factor:if update successful, increase step size by factor
    decrease_factor: if update unsuccessful, decrease step size by factor
    max_iters: max iterations (shouldnt need more than 5000 on noiseless testing func)
    tol: stopping tolerance of gradient

    Outputs:
    x: best point
    history: list of objective function vals over iterations
    """
    x = x0.astype(float)
    n = len(x)
    steps = np.ones(n) * step0
    history = [f(x)]
    
    for k in range(max_iters):
        g = grad(x)

        if np.linalg.norm(g) < tol:
            break

        idx = k % n

        #try update along coordinate
        x_new = x.copy()
        x_new[idx] -= steps[idx] * g[idx]
        
        if f(x_new) < f(x):  #improvement
            x = x_new
            steps[idx] *= increase_factor
        else: #not improvement
            steps[idx] *= decrease_factor

        history.append(f(x))
    
    return x, history

def rosenbrock4(x):
    x = np.asarray(x)
    return np.sum(100*(x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
def rosenbrock4_grad(x):
    x = np.asarray(x)
    n = len(x)
    g = np.zeros_like(x)

    # internal indices
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
def f(x):
    return (x[0]-3)**2 + (x[1]+1)**2

def grad(x):
    return np.array([2*(x[0]-3), 2*(x[1]+1)])

def numerical_grad(x):
    step=1e-6
    g = np.zeros_like(x)
    for i in range(len(x)):
        dx = np.zeros_like(x)
        dx[i] = step
        g[i] = (rosenbrock4(x + dx) - rosenbrock4(x - dx)) / (2*step)
    return g

if __name__ == "__main__":


    x0 = np.array([0,2.1,-22,20])
    solution, hist = adaptive_coordinate_descent(rosenbrock4, numerical_grad, x0)

    print("Solution:", solution)
    print("Final objective:", hist[-1])
