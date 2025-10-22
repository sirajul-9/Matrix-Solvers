#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 4   //Define dimension of the system (A is N×N, b and x are of length N)

/********************************************************************************/
/* Function: sor
 * ----------------------
 * Solves the linear system A x = b using the Successive Over-Relaxation (SOR) method.
 *
 * Parameters:
 *   a : coefficient matrix A (size N×N)
 *   b : right-hand side vector (size N)
 *   x : initial guess vector (is updated to the final solution)
 *   w : relaxation factor (0 < w <= 2, w=1 gives Gauss-Seidel)
 */
 
void sor(float a[][N], float *b, float *x, float w) {
    float sum, err = 1.0, e;
    int i, j, iter = 0;
    float x_new[N] = {0.0};

	// Iterate until convergence or maximum iteration limit is reached
    while (err > 1e-6 && iter < 100000) {
        iter++;

       for (i = 0; i < N; i++) {
            sum = 0.0;
            for (j = 0; j < N; j++) {
                if (j < i)
                    sum += a[i][j] * x_new[j];   // use updated values
                else if (j > i)
                    sum += a[i][j] * x[j];       // use old values
            }
            // SOR formula:
            x_new[i] = (1 - w) * x[i] + (w / a[i][i]) * (b[i] - sum);
        }

        // Compute maximum absolute difference
        e = fabs(x[0] - x_new[0]);
        for (i = 1; i < N; i++) {
            float diff = fabs(x[i] - x_new[i]);
            if (diff > e) e = diff;
        }
        err = e;

        // Update x
        for (i = 0; i < N; i++)
            x[i] = x_new[i];
    }

    printf("Total number of iterations = %d\n", iter);
    printf("Final error = %e\n", err);
}

/********************************************************************************/
int main() {
    int i, j;
    float A[N][N], b[N], x[N];
    float w = 1.3;  //  relaxation factor

    // Open files
    FILE *fp1 = fopen("mat.txt", "r");
    FILE *fp2 = fopen("vec.txt", "r");
    if (!fp1 || !fp2) {
        perror("Error opening file");
        return 1;
    }

    // Read matrix A
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            if (fscanf(fp1, "%f", &A[i][j]) != 1) {
                fprintf(stderr, "Error reading A[%d][%d]\n", i, j);
                return 1;
            }
        }
    }
    fclose(fp1);

    // Read vector b and initialize x
    for (i = 0; i < N; i++) {
        if (fscanf(fp2, "%f", &b[i]) != 1) {
            fprintf(stderr, "Error reading b[%d]\n", i);
            return 1;
        }
        x[i] = 0.0;
    }
    fclose(fp2);

    // Solve using SOR
    sor(A, b, x, w);

    // Write solution to sol.txt
    FILE *fp3 = fopen("sol.txt", "w");
    if (!fp3) {
        perror("Error opening sol.txt for writing");
        return 1;
    }
    for (i = 0; i < N; i++)
        fprintf(fp3, "%.8f\n", x[i]);
    fclose(fp3);

    return 0;
}

