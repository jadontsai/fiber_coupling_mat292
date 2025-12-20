import numpy as np
#file for nedler mead simplex algorithm. As detailed in the report, the function does derivative free 
#optimization by creating a simplex and iteratively updating points
def nelder_mead(f,x0, step=1, max_iter=50000, tol=1e-2, alpha=1.0, gamma=1.5, rho=0.5,sigma=0.5):
    """   
    Inputs (all floats unless otherwise specified):
        f: callable objective function (in this case power function). accepts 1D np array and returns float
        x0: initial guess of optimization variables(1D numpy array)
        step: initial simplex step size
        max_iter: maximum iterations (integer)
        tol: stopping condition. optimization stops when the simplex size falls below this threshold.
        alpha: reflection coefficient (typically around 1, can play around as long as >0)
        gamma: expansion coefficient  (typically around 2, can play around as long as >1)
        rho: contraction coefficient (typically 0.5 can play around as long as <1)
        sigma: shrink coefficient (typically 0.5, can play around as long as <1)
    Outputs:
        best_point: np array with coordinates for point of highest power
        best_value: Objective function value at best_point
    """

    n = len(x0) #number of dimensions, in this case 4
    
    #####initializing simplex
    simplex = np.zeros((n + 1, n)) #initialize, first vertex is initial guess
    simplex[0] = x0
    for i in range(n):
        y = np.array(x0, copy=True) #copy array into a y
        y[i] = y[i] + step
        simplex[i + 1] = y
    
    # Evaluate function at vertices
    f_vals = np.array([f(x) for x in simplex])
    
    for _ in range(max_iter):

        #Order simplex by function values
        idx = np.argsort(f_vals) #built in sorting function
        simplex = simplex[idx]
        f_vals = f_vals[idx]
        
        #Best, worst
        x_best = simplex[0]
        x_worst = simplex[-1]

        # Find centroid of shape that includes all but worst vertex
        centroid = np.mean(simplex[:-1], axis=0)
        #stop if stopping condition is met
        if np.max(np.linalg.norm(simplex - centroid, axis=1)) < tol:
            break
        #Reflection
        x_reflect = centroid + alpha * (centroid - x_worst)
        f_reflect = f(x_reflect)
         #do reflection if it improves over the worst but not best
        if f_vals[0] <= f_reflect < f_vals[-2]:
            simplex[-1] = x_reflect
            f_vals[-1] = f_reflect
            continue

        #Expansion (if reflection is better than curr best)
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
        #contraction
        if f_reflect < f_vals[-1]:
            # Outside contraction
            x_contract = centroid + rho * (x_reflect - centroid)
        else:
            # Inside contraction
            x_contract = centroid + rho * (x_worst - centroid)

        f_contract = f(x_contract)

        if f_contract < f_vals[-1]:
            simplex[-1] = x_contract
            f_vals[-1] = f_contract
            continue

        #Shrink (move all vertices to best point)
        for i in range(1, n + 1):
            simplex[i] = x_best + sigma * (simplex[i] - x_best)
            f_vals[i] = f(simplex[i])

    return simplex[0], f_vals[0]

