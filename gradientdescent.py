from __future__ import annotations
import math
import random
import time
from typing import Callable, Tuple, List, Optional
from dataclasses import dataclass

import numpy as pd
#helpers
tau = 2*math.pi

def wrap_angles(theta: pd.ndarray) -> pd.ndarray:
    # makes things periodic bc they can rotate more than 180 deg
    wrapped = (theta + math.pi) % tau - math.pi
    return wrapped

def central_diff_grad(f: Callable[[pd.ndarray], float], x: pd.ndarray,h: float = 1e-2, periodic: bool = True,) -> pd.ndarray:
#central difference gradient calculation (definition of derivative ish)
  #h: step
  #f: function
  #x: known angles
  #: periodic duh
    grad = pd.zeros_like(x, dtype=float)
    for i in range(len(x)):
        ei = pd.zeros_like(x)
        ei[i] = 1.0
        x_plus = x + h * ei
        x_minus = x - h * ei
        if periodic:
            x_plus = wrap_angles(x_plus)
            x_minus = wrap_angles(x_minus)
        f_plus = f(x_plus)
        f_minus = f(x_minus)
        grad[i] = (f_plus - f_minus) / (2.0 * h)
    return grad
#mathhhhhhhhhhhhhhh

@dataclass
class AscentResult:
    x_best: pd.ndarray #best found angles
    f_best: float #best result
    n_iter: int #number of iterations
    n_eval: int #number of evaluations
    history: List[Tuple[int, float]] #past result values
    converged: bool #did it converge orrr


def backtracking_line_search(
    f: Callable[[pd.ndarray], float],#objective
    x: pd.ndarray, #curr x value
    direction: pd.ndarray, #gradient direction 
    f_x: float, #curr val
    alpha: float = 1.0, #step size
    beta: float = 0.5, #change in step size
    c: float = 1e-4, #armijo constant
    max_trials: int = 20, #duh
    periodic: bool = True, #duh
) -> Tuple[pd.ndarray, float, int]:
    #Normalize direction to avoid huge steps if grad is large
    d_norm = pd.linalg.norm(direction)
    if d_norm == 0 or not pd.isfinite(d_norm):
        return x, f_x, 0
    d = direction / d_norm

    grad_proj = d_norm  #since direction is gradient, projection of gradient is the norm
    trials = 0
    a = alpha
    while trials < max_trials:
        x_new = x + a * d
        if periodic:
            x_new = wrap_angles(x_new)
        f_new = f(x_new)
        trials += 1
        # Armijo condition for ascent (apparenty): f(x_new) >= f(x) + c * a * ||grad||^2
        if f_new >= f_x + c * a * (grad_proj ** 2):
            return x_new, f_new, trials
        a *= beta
    return x, f_x, trials

def gradient_ascent(
    f: Callable[[pd.ndarray], float],
    x0: pd.ndarray,
    step0: float = 1.0,
    max_iter: int = 300,
    tol_grad: float = 1e-5,
    tol_f: float = 1e-9,
    h: float = 5e-3,
    line_search: bool = True,
    verbose: bool = False,
    periodic: bool = True,
) -> AscentResult:
    x = wrap_angles(pd.array(x0, dtype=float)) if periodic else pd.array(x0, dtype=float)
    f_x = f(x)
    n_eval = 1
    history = [(0, f_x)]
    converged = False

    step = float(step0)

    for it in range(1, max_iter + 1):
        g = central_diff_grad(f, x, h=h, periodic=periodic)
        n_eval += 2 * len(x)
        g_norm = float(pd.linalg.norm(g))

        if verbose:
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
                max_trials=20,
                periodic=periodic,
            )
            n_eval += trials  # each trial evaluates f once
            # If line search failed to improve, reduce step and try a small step
            if f_new <= f_x:
                step *= 0.5
                x_new = x + step * (g / (g_norm + 1e-12))
                if periodic:
                    x_new = wrap_angles(x_new)
                f_new = f(x_new)
                n_eval += 1
        else:
            x_new = x + step * (g / (g_norm + 1e-12))
            if periodic:
                x_new = wrap_angles(x_new)
            f_new = f(x_new)
            n_eval += 1

        # Check progress
        if f_new - f_x < tol_f:
            # not much improvement — consider converged
            converged = True
            x, f_x = x_new, f_new
            history.append((it, f_x))
            break

        x, f_x = x_new, f_new
        history.append((it, f_x))

        # Optional: adapt step based on success
        if line_search and len(history) >= 2 and history[-1][1] > history[-2][1]:
            step = min(step * 1.2, 5.0)

    return AscentResult(x_best=x, f_best=float(f_x), n_iter=it, n_eval=n_eval, history=history, converged=converged)
#testingggggggggggggggggggggggggggggggg

def optimize_angles(
    measure_fn: Callable[[pd.ndarray], float],
    x0: Optional[pd.ndarray] = None,
    n_vars: int = 4,
    step0: float = 0.8,
    max_iter: int = 300,
    tol_grad: float = 1e-4,
    tol_f: float = 1e-8,
    h: float = 5e-3,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> AscentResult:
    #for testing, otherwise just call gradient ascent function
    if seed is not None:
        random.seed(seed)
        pd.random.seed(seed)

    if x0 is None:
        x0 = pd.zeros(n_vars, dtype=float)
    else:
        x0 = pd.asarray(x0, dtype=float)
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
        verbose=verbose,
        periodic=True,
    )


def multistart_optimize(
    measure_fn: Callable[[pd.ndarray], float],
    n_starts: int = 8,
    n_vars: int = 4,
    step0: float = 0.8,
    max_iter: int = 300,
    tol_grad: float = 1e-4,
    tol_f: float = 1e-8,
    h: float = 5e-3,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> AscentResult:
    #same as optimize angles but with rando starts
    if seed is not None:
        random.seed(seed)
        pd.random.seed(seed)

    best: Optional[AscentResult] = None

    for k in range(n_starts):
        x0 = (pd.random.rand(n_vars) - 0.5) * tau  # (-π, π]
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
            verbose=verbose,
        )
        if best is None or res.f_best > best.f_best:
            best = res
        if verbose:
            print(f"[start {k+1}/{n_starts}] best f={best.f_best:.6g}")
    assert best is not None
    return best

def measure_coupling(angles: pd.ndarray) -> float:
#change later to actual values, this is just a vibecoded placeholder!!!!!!!!!!!
    # Synthetic "true" optimum near some offsets (unknown to optimizer)
    offsets = pd.array([0.7, -1.1, 0.4, 2.2])
    # Peaks shaped by cosines with different harmonics; add mild cross-terms
    val = (
        0.6 * pd.cos(angles[0] - offsets[0])
        + 0.9 * pd.cos(2.0 * (angles[1] - offsets[1]))
        + 0.7 * pd.cos(angles[2] - offsets[2])
        + 1.0 * pd.cos(3.0 * (angles[3] - offsets[3]))
    )
    val += 0.15 * pd.cos((angles[0] - angles[2])) + 0.1 * pd.cos((angles[1] + angles[3]))
    # Shift/scale to be positive
    return float(val + 4.0)


def _demo():
    print("somehow worked")
    t0 = time.time()
    result = multistart_optimize(
        measure_fn=measure_coupling,
        n_starts=12,
        n_vars=4,
        step0=0.8,
        max_iter=300,
        tol_grad=1e-5,
        tol_f=1e-9,
        h=5e-3,
        seed=42,
        verbose=False,
    )
    t1 = time.time()
    print(f"Converged: {result.converged}  iters: {result.n_iter}  evals: {result.n_eval}  time: {t1 - t0:.3f}s")
    print("Best angles (rad):", result.x_best)
    print("Best coupling:    ", result.f_best)


if __name__ == "__main__":
    _demo()

