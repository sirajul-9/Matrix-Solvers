/*
 * Conjugate Gradient Method (CG)
 * Solves A x = b for symmetric positive-definite A
 */


#include<stdio.h>
#include<math.h>
#include<time.h>
#include<stdlib.h>
#include"funcs.h"

#define N 4        // Matrix dimension


int main(){
    clock_t tic = clock(); //start time
    int i,j,flag=1,iter=0;
    double *A,*At,*b,*x,*xn,*r,*rn,*dum1,*p;
    double e=0.000001,alpha,beta;
    
    
    // --- Allocate memory for all vectors and matrices ---
    A =(double *)malloc(N*N * sizeof(double));
    At =(double *)malloc(N*N * sizeof(double));
    b = (double *)malloc(N*sizeof(double));
    x = (double *)malloc(N*sizeof(double));
    xn = (double *)malloc(N*sizeof(double));
    r = (double *)malloc(N*sizeof(double));
    rn = (double *)malloc(N*sizeof(double));
    dum1 = (double *)malloc(N*sizeof(double));
    p = (double *)malloc(N*sizeof(double));


     // --- Read matrix A and vector b from file ---
    read_2d("mat.txt",A,N);
    read_1d("vec.txt",b,N);
    
    // --- Ensure A is symmetric: A = (A + A^T)/2 ---
    transpose(A,At,N);
    for(i=0;i<N;i++)
    	for(j=0;j<N;j++)
    		A[i*N+j]=(A[i*N+j]+At[i*N+j])/2;

    // --- Initialize x = 0 ---
    for(i=0;i<N;i++){
        x[i]=0;
    }  
     
     
    // --- Compute initial residual r = b - A*x ---
    mul_mat_vec(A,x,dum1,N);
    for(i=0;i<N;i++){   
        r[i] = b[i] - dum1[i];
        p[i]=r[i];            // Initial search direction p0 = r0
    } 
    

    // --- Main Conjugate Gradient iteration loop ---
    while(flag==1){
        iter+=1;
        flag=0;
        
        // Compute A*p and step size α = (r^Tr) / (p^T A p)
        mul_mat_vec(A,p,dum1,N);
        alpha=mul_vec_vec(r,r,N)/mul_vec_vec(p,dum1,N);
        
        
        // x_{k+1} = x_k + alpha * p_k
        // r_{k+1} = r_k - alpha * A * p_k
        for(i=0;i<N;i++){   
            xn[i] = x[i]+alpha*p[i];
            rn[i] = r[i]-alpha*dum1[i];
        } 
        
        // Compute beta = (r_{k+1}^T * r_{k+1}) / (r_k^T * r_k)
        beta=mul_vec_vec(rn,rn,N)/mul_vec_vec(r,r,N);
        
        // Update search direction: p_{k+1} = r_{k+1} + beta * p_k
        for(i=0;i<N;i++)
            p[i]=rn[i]+beta*p[i];
            
        // Check for convergence
        for(i=0;i<N;i++){
            if(fabs(x[i]-xn[i])>e)
                flag=1;
        }
        
        // Update x and r for next iteration
        for(i=0;i<N;i++){
	    	x[i]=xn[i]; 
        	r[i]=rn[i]; 
        }
    }
    printf("Total number of iterations %d\n",iter);
    
    // --- Write outputs to a file ---
    FILE *fp = fopen("solution_cg.txt", "w");
	fprintf(fp, "Total iterations: %d\n", iter);
	fprintf(fp, "Final solution vector x:\n");
	for (i = 0; i < N; i++)
    	fprintf(fp, "%lf\n", x[i]);
	fclose(fp);
	
    clock_t toc = clock(); //end time
    
    // --- Free allocated memory ---
    free(A);
    free(At);
    free(b);
    free(r);
    free(p);
    free(dum1);
    free(x);
    free(xn);
    free(rn);

    printf("Elapsed: %lf seconds\n", (double)(toc - tic) / CLOCKS_PER_SEC);
    return 0;
}
    
