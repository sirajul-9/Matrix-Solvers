/*
 * Minimal Residual Method (CG)
 * Solves A x = b for symmetric A (positive definiteness not necessary)
 */

#include<stdio.h>
#include<math.h>
#include<time.h>
#include<stdlib.h>
#include"funcs.h"

#define N 4     // dimension of the matrix


int main(){
    clock_t tic = clock();
    int i,j,flag=1,iter=0;
    double *A,*At,*b,*x,*xn,*r,*dum1,*p;
    double e=0.000001,alpha;
    
    
    // --- Allocating memory for all matrices and vectors
    A =(double *)malloc(N*N * sizeof(double));
    At =(double *)malloc(N*N * sizeof(double));
    b = (double *)malloc(N*sizeof(double));
    x = (double *)malloc(N*sizeof(double));
    xn = (double *)malloc(N*sizeof(double));
    r = (double *)malloc(N*sizeof(double));
    dum1 = (double *)malloc(N*sizeof(double));
    p = (double *)malloc(N*sizeof(double));


    // --- reading the matrix A and vector b from file ---
    read_2d("mat.txt",A,N);
    read_1d("vec.txt",b,N);
    
    //--- ensure A is symmetric ---
    transpose(A, At, N);
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            A[i*N + j] = 0.5 * (A[i*N + j] + At[i*N + j]);
        }
    }

    //--- initialising with zero---
    for(i=0;i<N;i++){
        x[i]=0;
    }  
     
    // --- Compute initial residual r = b - A*x ---
    mul_mat_vec(A,x,dum1,N);
    for(i=0;i<N;i++){   
        r[i] = b[i] - dum1[i];
    } 
    
    // compute p = A*r
    mul_mat_vec(A,r,p,N);
    
    // ---- MINRES itertaion loop ---- //
    while(flag==1){
        iter+=1;
        flag=0;
        
        // alpha = (rᵀ A r) / ((A r)ᵀ (A r))
        alpha=mul_vec_vec(r,p,N)/mul_vec_vec(p,p,N);
        
        // Update solution and residual:
        // x_{k+1} = x_k + α * r_k
        // r_{k+1} = r_k - α * A * r_k
        for(i=0;i<N;i++){   
            xn[i] = x[i]+alpha*r[i];
            r[i] =r[i] - alpha*p[i];
        } 
        
        // Compute new p = A * r
        mul_mat_vec(A,r,p,N);
        
        // Update x for next iteration
        for(i=0;i<N;i++){
            if(fabs(x[i]-xn[i])>e)
                flag=1;
        }
        
         // Update x for next iteration
        for(i=0;i<N;i++)
	        x[i]=xn[i]; 

    }
    
   
    /* ------------------ Write Output ------------------ */
    FILE *fout = fopen("sol_minres.txt", "w");
    fprintf(fout, "------------------------------------------\n");
    fprintf(fout, "        MINRES Iterative Solver\n");
    fprintf(fout, "------------------------------------------\n");
    fprintf(fout, "Matrix dimension: %d\n", N);
    fprintf(fout, "Total iterations: %d\n", iter);
    fprintf(fout, "\nFinal solution vector x:\n");
    for (i = 0; i < N; i++)
        fprintf(fout, "%lf\n", x[i]);

    clock_t toc = clock();
    double elapsed = (double)(toc - tic) / CLOCKS_PER_SEC;

    fprintf(fout, "\nElapsed time: %lf seconds\n", elapsed);
    fclose(fout);

    printf("MINRES completed in %d iterations (%.6lf s)\n", iter, elapsed);
    printf("Results written to sol_minres.txt\n");


	// --- clean-up allocated memory ---
    free(A);
    free(b);
    free(r);
    free(p);
    free(dum1);
    free(x);
    free(xn);
    free(At);
    
    return 0;
}
    
