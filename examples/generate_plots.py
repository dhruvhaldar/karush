import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from karush.unconstrained.newton import newton_method
from karush.unconstrained.quasi_newton import bfgs_method
from karush.unconstrained.conjugate_gradient import conjugate_gradient
from karush.constrained.sqp import sqp_equality_constrained

def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def rosenbrock_grad(x):
    g = np.zeros_like(x)
    g[0] = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
    g[1] = 200 * (x[1] - x[0]**2)
    return g

def rosenbrock_hess(x):
    H = np.zeros((2, 2))
    H[0, 0] = 2 - 400 * x[1] + 1200 * x[0]**2
    H[0, 1] = -400 * x[0]
    H[1, 0] = -400 * x[0]
    H[1, 1] = 200
    return H

def plot_unconstrained():
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-1, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = rosenbrock(np.array([X[i, j], Y[i, j]]))

    plt.figure(figsize=(10, 8))
    # Log scale for contours
    plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='viridis')
    plt.plot(1, 1, 'r*', markersize=15, label='Optimum (1,1)')
    
    x0 = [-1.2, 1.0]
    
    # Newton
    try:
        x_opt, path_newton = newton_method(rosenbrock, rosenbrock_grad, rosenbrock_hess, x0)
        path_newton = np.array(path_newton)
        plt.plot(path_newton[:, 0], path_newton[:, 1], 'o-', label='Newton')
    except Exception as e:
        print(f"Newton failed: {e}")
    
    # BFGS
    try:
        x_opt, path_bfgs = bfgs_method(rosenbrock, rosenbrock_grad, x0)
        path_bfgs = np.array(path_bfgs)
        plt.plot(path_bfgs[:, 0], path_bfgs[:, 1], 's-', label='BFGS')
    except Exception as e:
        print(f"BFGS failed: {e}")
    
    # CG
    try:
        x_opt, path_cg = conjugate_gradient(rosenbrock, rosenbrock_grad, x0)
        path_cg = np.array(path_cg)
        plt.plot(path_cg[:, 0], path_cg[:, 1], '^-', label='Conjugate Gradient')
    except Exception as e:
        print(f"CG failed: {e}")
    
    plt.legend()
    plt.title('Unconstrained Optimization on Rosenbrock Function')
    plt.xlabel('x')
    plt.ylabel('y')
    
    output_path = os.path.join(os.path.dirname(__file__), '../docs/images/unconstrained.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")

def plot_constrained():
    # Min x^2 + y^2 s.t. x + y = 1
    # Global min at (0.5, 0.5)
    
    def f(x): return x[0]**2 + x[1]**2
    def grad_f(x): return np.array([2*x[0], 2*x[1]])
    def hess_f(x): return np.array([[2, 0], [0, 2]])
    def h(x): return np.array([x[0] + x[1] - 1])
    def grad_h(x): return np.array([1, 1])
    
    x = np.linspace(-0.5, 1.5, 100)
    y = np.linspace(-0.5, 1.5, 100)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2
    
    plt.figure(figsize=(8, 8))
    plt.contour(X, Y, Z, levels=20, cmap='viridis')
    
    # Constraint line x + y = 1 => y = 1 - x
    plt.plot(x, 1 - x, 'k--', label='Constraint x+y=1')
    
    x0 = [0.0, 0.0] # Start at origin
    
    try:
        x_opt, path_sqp = sqp_equality_constrained(f, grad_f, hess_f, h, grad_h, x0)
        path_sqp = np.array(path_sqp)
        plt.plot(path_sqp[:, 0], path_sqp[:, 1], 'mo-', label='SQP Path')
    except Exception as e:
        print(f"SQP failed: {e}")

    plt.plot(0.5, 0.5, 'r*', markersize=15, label='Optimum')
    
    plt.legend()
    plt.title('Equality Constrained Optimization (SQP)')
    plt.xlabel('x')
    plt.ylabel('y')
    
    output_path = os.path.join(os.path.dirname(__file__), '../docs/images/constrained.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")

if __name__ == "__main__":
    plot_unconstrained()
    plot_constrained()
