#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define N 4   //Define dimension of the system (A is N×N, b and x are length N)



/********************************************************************************/ 
/* Function: jacobi
 * ----------------
 * Solves the linear system A x = b using the Jacobi iterative method.
 * 
 * Parameters:
 *   a : coefficient matrix A (size N×N)
 *   b : right-hand side vector (size N)
 *   x : initial guess vector (is updated to the final solution)
 */
 
void jacobi(float a[][N], float *b, float *x) {
    float sum, err = 1.0, e;
    int i, j, iter = 0;
    float x_new[N] = {0};  // Temporary array to hold updated values of x

    // Iterate until convergence or maximum iteration limit is reached
    while (err > 1e-6 && iter < 100000) {
        iter++;

        // Compute new estimates x_new[i] for all i using previous x values
        for (i = 0; i < N; i++) {
            sum = 0.0;
            for (j = 0; j < N; j++) {
                if (i != j)
                    sum += a[i][j] * x[j];   // considering off-diagonal terms
            }
            x_new[i] = (b[i] - sum) / a[i][i];  // Jacobi update formula
        }

        // Compute maximum absolute difference between old and new x
        e = fabs(x[0] - x_new[0]);
        for (i = 1; i < N; i++) {
            float diff = fabs(x[i] - x_new[i]);
            if (diff > e)
                e = diff;
        }
        err = e;   // Update current error estimate

        // Copy x_new into x for next iteration
        for (i = 0; i < N; i++)
            x[i] = x_new[i];
    }

    // Display number of iterations and final error
    printf("Total number of iterations = %d\n", iter);
    printf("Final error = %e\n", err);
}
/***********************************************************************/



int main() {
    float A[N][N], b[N], x[N];
    int i, j;

    // Open files containing matrix A and vector b
    FILE *fp1 = fopen("mat.txt", "r");
    FILE *fp2 = fopen("vec.txt", "r");

    // Check for file open errors
    if (!fp1 || !fp2) {
        perror("Error opening file");
        return 1;
    }

    // Read matrix A from mat.txt
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            if (fscanf(fp1, "%f", &A[i][j]) != 1) {
                fprintf(stderr, "Error reading A[%d][%d]\n", i, j);
                return 1;
            }
        }
    }
    fclose(fp1);  // Close matrix file

    // Read vector b from vec.txt and initialize x = 0
    for (i = 0; i < N; i++) {
        if (fscanf(fp2, "%f", &b[i]) != 1) {
            fprintf(stderr, "Error reading b[%d]\n", i);
            return 1;
        }
        x[i] = 0.0;  //setting intial guess solution
    }
    fclose(fp2);  // Close vector file

    // Solve the system using Jacobi iteration
    jacobi(A, b, x);

    // Write the computed solution vector x to sol.txt
    FILE *fp3 = fopen("sol.txt", "w");
    for (i = 0; i < N; i++)
        fprintf(fp3, "%f\n", x[i]);
    fclose(fp3);

    return 0;
}

