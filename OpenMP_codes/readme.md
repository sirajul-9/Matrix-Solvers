## 🚀 How to Compile and Run

Each solver is standalone and can be compiled using GCC:
### Jacobi
```bash
gcc -o jacobi jacobi_openmp.c -lm -fopenmp
./jacobi
```
### Conjugate Gradient
```bash
gcc -o cg cg_openmp.c functions_omp.c -lm -fopenmp
./cg
```
### Bi-conjugate Gradient Stabilized
```bash
gcc -o bicgstab bicgstab_openmp.c functions_omp.c -lm -fopenmp
./bicgstab
```
