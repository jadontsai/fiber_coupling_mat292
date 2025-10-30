import math
import random
from typing import Callable, Tuple, List, Optional
from dataclasses import dataclass

import numpy as np
#helpers
tau = 2*math.pi


def central_diff_grad(f: Callable[[np.ndarray], float], x: np.ndarray,h: float = 1e-2):
#central difference gradient calculation (definition of derivative ish)
  #h: step
  #f: function
  #x: known angles
    grad = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        ei = np.zeros_like(x)
        ei[i] = 1.0
        x_plus = x + h * ei
        x_minus = x - h * ei
        f_plus = f(x_plus)
        f_minus = f(x_minus)
        grad[i] = (f_plus - f_minus) / (2.0 * h)
    return grad

@dataclass
class AscentResult:
    x_best: np.ndarray #best found angles
    f_best: float #best result
    n_iter: int #number of iterations
    n_eval: int #number of evaluations
    history: List[Tuple[int, float]] #past result values
    converged: bool #did it converge


def backtracking_line_search(
    f: Callable[[np.ndarray], float],#objective
    x: np.ndarray, #curr x value
    direction: np.ndarray, #gradient direction 
    f_x: float, #curr val
    alpha: float = 1.0, #step size
    beta: float = 0.5, #change in step size
    c: float = 1e-4, #armijo constant
    max_trials: int = 20, #duh
):
    #Normalize direction to avoid huge steps if grad is large
    d_norm = np.linalg.norm(direction)
    if d_norm == 0 or not np.isfinite(d_norm):
        print("done or broken")
        return x, f_x, 0
    d = direction / d_norm

    grad_proj = d_norm  #since direction is gradient, projection of gradient is the norm
    trials = 0
    a = alpha
    while trials < max_trials:
        x_new = x + a * d
        f_new = f(x_new)
        trials += 1
        # Armijo condition for ascent (apparenty): f(x_new) >= f(x) + c * a * ||grad||^2
        if f_new >= f_x + c * a * (grad_proj ** 2):
            return x_new, f_new, trials
        a *= beta
    return x, f_x, trials

def gradient_ascent(
    f: Callable[[np.ndarray], float],
    x0: np.ndarray,
    step0: float = 1.0,
    max_iter: int = 300,
    tol_grad: float = 1e-5,
    tol_f: float = 1e-9,
    h: float = 5e-3,
    line_search: bool = True,
    need_details: bool = True
):
    x = np.array(x0, dtype=float)
    f_x = f(x)
    n_eval = 1
    history = [(0, f_x)]
    converged = False

    step = float(step0)

    for it in range(1, max_iter + 1):
        g = central_diff_grad(f, x, h=h)
        n_eval += 2 * len(x)
        g_norm = float(np.linalg.norm(g))

        if need_details:
            print(f"[iter {it}] f={f_x:.6g} ||g||={g_norm:.3e} step={step:.3e}")

        if g_norm < tol_grad:
            converged = True
            break

        if line_search:
            x_new, f_new, trials = backtracking_line_search(
                f=f,
                x=x,
                direction=g,
                f_x=f_x,
                alpha=step,
                beta=0.5,
                c=1e-4,
                max_trials=20
            )
            n_eval += trials  # each trial evaluates f once
            # If line search failed to improve, reduce step and try a small step
            if f_new <= f_x:
                step *= 0.5
                x_new = x + step * (g / (g_norm + 1e-12))
                f_new = f(x_new)
                n_eval += 1
        else:
            x_new = x + step * (g / (g_norm + 1e-12))
            f_new = f(x_new)
            n_eval += 1

        # Check progress
        if f_new - f_x < tol_f:
            converged = True
            x, f_x = x_new, f_new
            history.append((it, f_x))
            break

        x, f_x = x_new, f_new
        history.append((it, f_x))

        #also may remove later
        if line_search and len(history) >= 2 and history[-1][1] > history[-2][1]:
            step = min(step * 1.2, 5.0)

    return AscentResult(x_best=x, f_best=float(f_x), n_iter=it, n_eval=n_eval, history=history, converged=converged)
#testingggggggggggggggggggggggggggggggg

def optimize_angles(
    measure_fn: Callable[[np.ndarray], float],
    x0: Optional[np.ndarray] = None,
    n_vars: int = 4,
    step0: float = 0.8,
    max_iter: int = 300,
    tol_grad: float = 1e-4,
    tol_f: float = 1e-8,
    h: float = 5e-3,
    seed: Optional[int] = None,
    need_details: bool = True,
) -> AscentResult:
    #for testing, otherwise just call gradient ascent function
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if x0 is None:
        x0 = np.zeros(n_vars, dtype=float)
    else:
        x0 = np.asarray(x0, dtype=float)
        if x0.shape != (n_vars,):
            raise ValueError(f"x0 must have shape ({n_vars},), got {x0.shape}")

    return gradient_ascent(
        f=measure_fn,
        x0=x0,
        step0=step0,
        max_iter=max_iter,
        tol_grad=tol_grad,
        tol_f=tol_f,
        h=h,
        line_search=True,
        need_details=need_details,
    )


def multistart_optimize(
    measure_fn: Callable[[np.ndarray], float],
    n_starts: int = 8,
    n_vars: int = 4,
    step0: float = 0.8,
    max_iter: int = 300,
    tol_grad: float = 1e-4,
    tol_f: float = 1e-8,
    h: float = 5e-3,
    seed: Optional[int] = None,
    need_details: bool = True,
):
    #same as optimize angles but with rando starts
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    best: Optional[AscentResult] = None

    for k in range(n_starts):
        x0 = (np.random.rand(n_vars) - 0.5) * tau #change later
        res = optimize_angles(
            measure_fn=measure_fn,
            x0=x0,
            n_vars=n_vars,
            step0=step0,
            max_iter=max_iter,
            tol_grad=tol_grad,
            tol_f=tol_f,
            h=h,
            seed=None,
            need_details=need_details,
        )
        if best is None or res.f_best > best.f_best:
            best = res
        if need_details:
            print(f"[start {k+1}/{n_starts}] best f={best.f_best:.6g}")
    assert best is not None
    return best

if __name__ == "__main__":
    pass
