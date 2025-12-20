import numpy as np
from power_function import objective #objective function
from power_function import power_history #for plotting
import matplotlib.pyplot as plt


def adaptive_coordinate_descent(
    f,
    x0,
    initial_step=0.1,
    step_min=1e-4,
    expand=1.3,
    shrink=0.5,
    max_iters=200
):
    """
    Performs adaptive coordinate descent on our imported power function
    Inputs (float unless otherwise noted):
    f : callable function
        Objective function to minimize; accepts np array and returns scalar
    x0: np array: initial guess
    initial_step: initial step size
    step_min: stopping condition - stops when step size smaller than this value
    expand: factor to increase a coordinate step size after an improvement
    shrink: factor to decrease step size
    max_iters: maximum number of iterations allowed (other stopping condition)

    Outputs:
    x: position of best vector
    best_value: max power
    """
    x = np.array(x0, dtype=float)

    #Number of parameters (i.e. number of dimension space, in our case 4)
    n = len(x)

    #one step size per coordinate
    steps = np.full(n, initial_step)

    #evaluate fcn
    f_best = f(x)

    for it in range(max_iters):
        improved_any = False  #track if any coordinate improved this iteration

        #Loop over coordinates
        for i in range(n):
            improved = False  #track improvement for this coordinate

            #try both positive and negative directions
            for direction in (+1, -1):
                x_try = x.copy()
                x_try[i] += direction * steps[i]
                # Evaluate objective at trial point
                f_try = f(x_try)
                
                if f_try < f_best:# If improvement found,
                    # do it
                    x = x_try
                    f_best = f_try

                    #increase step size bc improvement
                    steps[i] *= expand
                    improved = True
                    improved_any = True
                    break

            #If neither direction improved, reduce step size
            if not improved:
                steps[i] *= shrink

        #Stop if all step sizes are very small
        if np.max(steps) < step_min:
            print(f"Converged iteration number {it}")
            break

    return x, -f_best


x0 = np.array([0.67, 0.67, 2.1, 2.67])#initial guess

#run
best_x, best_power = adaptive_coordinate_descent(
    objective,
    x0,
    initial_step=0.2,
    max_iters=300
)

print("\nBest parameters found:", best_x)
print("Best power:", best_power)

power_history_arr = np.array(power_history) #for plotting

plt.figure()
plt.plot(power_history_arr, linewidth=1)
plt.xlabel("Iteration")
plt.ylabel("Measured power")
plt.title(f"Adaptive gradient descent for {x0}")
plt.grid(True)
plt.show()
