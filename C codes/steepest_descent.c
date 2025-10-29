#include<stdio.h>
#include<math.h>
#include<time.h>
#include<stdlib.h>
#include"funcs.h"    //contains some helper functions

#define N 4         // matrix dimension


int main(){
    clock_t tic = clock();       // Start timer
    int i,j,flag=1,iter=0;
    double *A,*At,*b,*x,*xn,*r,*dum1,*p;
    double e=0.000001,alpha;
    
    // ---- Memory Allocation -------
    A    = (double *)malloc(N * N * sizeof(double));  // coefficient matrix A
    At   = (double *)malloc(N * N * sizeof(double));  // Transpose of A
    b    = (double *)malloc(N * sizeof(double));      // RHS vector b
    x    = (double *)malloc(N * sizeof(double));      // Current solution
    xn   = (double *)malloc(N * sizeof(double));      // Updated solution
    r    = (double *)malloc(N * sizeof(double));      // Residual vector
    dum1 = (double *)malloc(N * sizeof(double));      // Temporary storage
    p    = (double *)malloc(N * sizeof(double));      // A * r (search direction)
    
    
    // ---- Read input files ----
    // Matrix (A) from mat.txt and vector (b) from vec.txt
    read_2d("mat.txt",A,N);
    read_1d("vec.txt",b,N);
    
    // ---- Symmetrize A ----
    transpose(A,At,N);
    for(i=0;i<N;i++)
    	for(j=0;j<N;j++)
    		A[i*N+j]=(A[i*N+j]+At[i*N+j])/2.0;
    
    // ---- Initial guess ----
    for(i=0;i<N;i++){
        x[i]=0;
    }  
   
    // ---- Compute initial residual r = b - A*x ----
    mul_mat_vec(A,x,dum1,N);
    for(i=0;i<N;i++){   
        r[i] = b[i] - dum1[i];
    } 
    mul_mat_vec(A,r,p,N);  // Compute p = A*r  (used for denominator in alpha)

	//--------- steepest descent iteration --------
    while(flag==1){
    
        iter+=1;		// count iterations
        flag=0;         // Reset flag; will be set to 1 if convergence not reached
        
        // step size: alpha = (rᵀ r) / (rᵀ A r)
        alpha=mul_vec_vec(r,r,N)/mul_vec_vec(r,p,N);
        
        // update solution and residual simultaneously
        for(i=0;i<N;i++){   
            xn[i] = x[i]+alpha*r[i];        // new solution: x_{k+1} = x_k + α r_k
            r[i] =r[i] - alpha*p[i];        // updated residual: r_{k+1} = r_k - α A r_k
        } 
        
        mul_mat_vec(A,r,p,N);               // compute new p = A * r for next iteration
        
        // check convergence: if any |x_new - x_old| > tolerance, keep iterating
        for(i=0;i<N;i++){
            if(fabs(x[i]-xn[i])>e)
                flag=1;
        }
        
        // copy new solution into x for the next iteration
        for(i=0;i<N;i++)
	    	x[i]=xn[i];  
    }
    
    //--output final solution ----
    printf("Total number of iterations %d\n",iter);
    printf("Final solution vector x:\n");
    for(i=0;i<N;i++)
    	printf("%lf\n",x[i]);
    
    //--calculate execution time--
    clock_t toc = clock();
    printf("Elapsed: %lf seconds\n", (double)(toc - tic) / CLOCKS_PER_SEC);
    
    //---free allocated memory----
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
    
