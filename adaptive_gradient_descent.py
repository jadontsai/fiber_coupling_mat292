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




