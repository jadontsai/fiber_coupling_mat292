import numpy as np

def nelder_mead(f,x0, step=1, max_iter=50000, tol=1e-2, alpha=1.0, gamma=1.5, rho=0.5,sigma=0.5):
    """    
    Inputs:
        f: objective function f(x)
        x0: initial guess (1D numpy array)
        step: initial simplex step size
        max_iter: maximum iterations
        tol: stopping tolerance
        alpha:
    Outputs:
        (best_point, best_value) tuple
    """

    n = len(x0) #number of dimensions, in this case 4
    
    #####initializing simplex
    simplex = np.zeros((n + 1, n)) #initialize 
    simplex[0] = x0
    for i in range(n):
        y = np.array(x0, copy=True) #copy array into a y
        y[i] = y[i] + step
        simplex[i + 1] = y
    
    # Evaluate initial simplex
    f_vals = np.array([f(x) for x in simplex])
    
    for _ in range(max_iter):

        #Order simplex by function values
        idx = np.argsort(f_vals) #built in sorting function
        simplex = simplex[idx]
        f_vals = f_vals[idx]
        
        #Best, worst
        x_best = simplex[0]
        x_worst = simplex[-1]

        #check if values are all within tolerance
        if np.std(f_vals) < tol:
            break

        # Find centroid of shape that includes all but worst vertex
        centroid = np.mean(simplex[:-1], axis=0) #works somehow, read documentation

        #Reflection
        x_reflect = centroid + alpha * (centroid - x_worst)
        f_reflect = f(x_reflect)

        if f_vals[0] <= f_reflect < f_vals[-2]: #i think this works?
            simplex[-1] = x_reflect
            f_vals[-1] = f_reflect
            continue

        #Expansion
        if f_reflect < f_vals[0]:
            x_expand = centroid + gamma * (x_reflect - centroid)
            f_expand = f(x_expand)
            if f_expand < f_reflect:
                simplex[-1] = x_expand
                f_vals[-1] = f_expand
            else:
                simplex[-1] = x_reflect
                f_vals[-1] = f_reflect
            continue

        #Contraction
        x_contract = centroid + rho * (x_worst - centroid)
        f_contract = f(x_contract)

        if f_contract < f_vals[-1]:
            simplex[-1] = x_contract
            f_vals[-1] = f_contract
            continue

        #Shrink
        for i in range(1, n + 1):
            simplex[i] = x_best + sigma * (simplex[i] - x_best)
            f_vals[i] = f(simplex[i])

    return simplex[0], f_vals[0]
