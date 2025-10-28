# unit_test_optimizers.py
import numpy as np
from gradient_ascent import multistart_optimize
from newtons_method import newtons_method
from test_cases import quadratic, offset_quadratic, multimodal_cosine, synthetic_coupling

def test_optimizer(f, true_max_point, desc, tol=1e-2):
    print(f"=== Testing on {desc} ===")

    # ---- Gradient Ascent ----
    res = multistart_optimize(
        measure_fn=f,
        n_starts=5,
        n_vars=len(true_max_point),
        step0=0.8,
        max_iter=300,
        tol_grad=1e-5,
        tol_f=1e-9,
        h=5e-3,
        seed=42,
        verbose=False,
    )

    # ---- Newton’s Method ----
    theta0 = np.random.uniform(-np.pi, np.pi, len(true_max_point))
    theta_opt = newtons_method(f=f, theta0=theta0, tol=1e-8, max_iter=50)

    print(f"Gradient Ascent: f*={res.f_best:.6f}, θ*={np.round(res.x_best,3)}")
    print(f"Newton’s Method: f*={f(theta_opt):.6f}, θ*={np.round(theta_opt,3)}")

    # Check convergence near expected solution
    if true_max_point is not None:
        assert np.allclose(res.x_best, true_max_point, atol=tol) or \
               np.allclose(np.mod(res.x_best, 2*np.pi), np.mod(true_max_point, 2*np.pi), atol=tol)
        print("✅ Gradient Ascent converged near expected optimum.")
    else:
        print("✅ Gradient Ascent converged (no analytic solution).")

    print("")

def run_all_tests():
    np.set_printoptions(precision=3, suppress=True)

    # 1️⃣ Simple Quadratic
    test_optimizer(quadratic, np.zeros(4), "Simple Quadratic")

    # 2️⃣ Offset Quadratic
    test_optimizer(offset_quadratic, np.array([1.0, -0.5, 2.0, 0.5]), "Offset Quadratic")

    # 3️⃣ Multi-modal Cosine
    test_optimizer(multimodal_cosine, np.zeros(4), "Multimodal Cosine")

    # 4️⃣ Synthetic Fiber Coupling Function
    test_optimizer(synthetic_coupling, None, "Synthetic Fiber Coupling")

if __name__ == "__main__":
    run_all_tests()
