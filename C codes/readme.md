# 🧮 Iterative Solvers for Linear Systems

This repository contains C implementations of some of the classic **iterative methods** for solving linear systems of equations of the form:

$$
\textbf{A} x = b
$$

These methods are particularly useful for large, sparse, or structured matrices where direct solvers (like Gaussian elimination) become computationally expensive.

---

## 📂 Repository Structure

| File | Description |
|------|--------------|
| **jacobi.c** | Implementation of the **Jacobi Iteration** method. |
| **gauss_seidel.c** | Implementation of the **Gauss–Seidel** method. |
| **successive_over-relaxation.c** | Implementation of the **SOR (Successive Over-Relaxation)** method. |
| **steepest_descent.c** | Implementation of the **Steepest Descent** method for SPD systems. |
| **conjugate_gradient.c** | Implementation of the **Conjugate Gradient (CG)** method — faster convergence for SPD matrices. |
| **functions.c** | contains some helper functions (matrix–vector multiplication, vector dot product, etc.) used by iterative solvers. |
| **funcs.h** | Header file containing function declarations. |
| **mat.txt** | Input file containing the system matrix \(A\) (row-wise format). |
| **vec.txt** | Input file containing the right-hand side vector \(b\). |
| **sol.txt** | Output file storing the computed solution vector. |
| **readme.md** | This documentation file. |

---

## 🚀 How to Compile and Run

Each solver is standalone and can be compiled using GCC:
### Jacobi
```bash
gcc -o jacobi jacobi.c -lm
./jacobi
```
### Gauss-Seidel
```bash
gcc -o gs gauss_seidel.c -lm
./gs
```
### Successive Over-relaxation
```bash
gcc -o sor successive_over-relaxation.c -lm
./sor
```
### Steepest Descent
```bash
gcc -o sd steepest_descent.c functions.c -lm
./sd
```
### Conjugate Gradient
```
gcc -o cg conjugate_gradient.c functions.c -lm
./cg
```

