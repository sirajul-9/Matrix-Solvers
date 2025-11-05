"""
Conjugate Gradient Method (CG)
------------------------------
Solves A x = b for symmetric positive-definite matrices A.

Files required:
    mat.txt : Matrix A (flattened or n×n text)
    vec.txt : RHS vector b
"""
import numpy as np
import time

# --- Start timer ---
start_time = time.time()

# --- Read matrix and vector from files ---
A = np.loadtxt("mat.txt")
b = np.loadtxt("vec.txt")

n=len(b)
A=A.reshape([n,n])

# --- Ensure A is symmetric: A = (A + Aᵀ)/2 ---
A=(A+A.T)/2

# --- Initialize variables ---
x = np.zeros(n)   # Initial guess x0 = 0
r = b - A @ x     # Initial residual: r0 = b - Ar0
p = r.copy()      # Initial search direction 
tol = 1e-6        # Convergence tolerance
error = 1.0
iterations = 0


# --- Main Conjugate Gradient iteration loop ---
while error > tol:
    iterations += 1
    
    # Compute A*p
    Ap = A @ p
    
    # Step size: alpha = (r.r) / (p.A.p)
    alpha = np.dot(r, r) / np.dot(p, Ap)
    
    # Update solution and residual
    xn = x + alpha * p
    rn = r - alpha * Ap
    
    # Compute beta = (rn.rn) / (r.r)
    beta = np.dot(rn, rn) / np.dot(r, r)

    # Update search direction: p = rn + beta x p
    p = rn + beta * p

    # Check for convergence
    error = np.max(np.abs(xn - x))

    # Prepare for next iteration
    x = xn
    r = rn

# --- End timer ---
end_time = time.time()
    
    
# --- Write results to file ---
with open("solution_cg_python.txt", "w") as f:
    f.write(f"Total iterations: {iterations}\n")
    f.write("Final solution vector x:\n")
    for xi in x:
        f.write(f"{xi:.10f}\n")
    f.write(f"\nElapsed time: {end_time - start_time:.6f} seconds\n")

# --- Print summary on screen ---
print(f"Total iterations: {iterations}")
print("Final solution vector x:\n", x)
print(f"Elapsed time: {end_time - start_time:.6f} seconds")
