#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<time.h>
#include"funcs.h"

#define N 4       // Matrix dimension


int main(){

    clock_t tic = clock();
    int i,j,flag=1,iter=0;
    double *A,*At,*b,*x,*xn,*r,*rn,*dum1,*p,*dum2,*r2,*s;
    double e=0.000001,alpha,beta,w;
    
    // --- Allocating memory for necessary matrices and vectors ---
    A = (double *)malloc(N * N * sizeof(double));
    At = (double *)malloc(N * N * sizeof(double));
    b = (double *)malloc(N * sizeof(double));
    x = (double *)malloc(N * sizeof(double));
    xn = (double *)malloc(N * sizeof(double));
    r = (double *)malloc(N * sizeof(double));
    rn = (double *)malloc(N * sizeof(double));
    dum1 = (double *)malloc(N * sizeof(double));
    dum2 = (double *)malloc(N * sizeof(double));
    p = (double *)malloc(N * sizeof(double));
    r2 = (double *)malloc(N * sizeof(double));
    s = (double *)malloc(N * sizeof(double));

	// --- Read matrix A and vector b from file ---
    read_2d("mat.txt",&A[0],N);
    read_1d("vec.txt",b,N);
    
	 // --- Initialize vectors ---
    for (i = 0; i < N; i++) {
        x[i] = 0.0;
    }
  
    // Compute initial residual r = b - A*x
    mul_mat_vec(&A[0],x,dum1,N);
    for(i=0;i<N;i++){   
        r[i] = b[i] - dum1[i];
        r2[i]=r[i];
    } 
    
    // Initial search direction p = r
    for (i = 0; i < N; i++)
        p[i] = r[i];

    while(flag==1){
        iter+=1;
        flag=0;
        
         // Compute A*p and step size alpha = (r, r2) / (A*p, r2)
        mul_mat_vec(&A[0],p,dum1,N);
        alpha=mul_vec_vec(r,r2,N)/mul_vec_vec(r2,dum1,N);
        
        // Compute s = r - alpha * A*p
        for(i=0;i<N;i++){
            s[i]=r[i]-alpha*dum1[i];
        }
        
        // Compute A*s and w
        mul_mat_vec(&A[0],s,dum2,N);
        w=mul_vec_vec(dum2,s,N)/mul_vec_vec(dum2,dum2,N);
        
        // Update x_{k+1} = x_k + alpha p + ws
        // Update r_{k+1} = s - wA*s
        for(i=0;i<N;i++){   
            xn[i] = x[i]+alpha*p[i]+w*s[i];
            rn[i] = s[i]-w*dum2[i];
        } 
        
        // Compute beta = ((r_{k+1}, r2) / (r_k, r2)) * (alpha / w)
        beta=mul_vec_vec(rn,r2,N)/mul_vec_vec(r,r2,N) * (alpha/w);
        
        // Update search direction p = r_{k+1} + alpha (p - wA*p)
        for(i=0;i<N;i++)
            p[i]=rn[i]+beta*(p[i]-w*dum1[i]);
            
        // Convergence check   
        for(i=0;i<N;i++){
            if(fabs(x[i]-xn[i])>e)
                flag=1;
        }
        
        // Prepare for next iteration
        for(i=0;i<N;i++){
	        x[i]=xn[i]; 
            r[i]=rn[i]; 
        }
    }
     printf("Total number of iterations %d\n", iter);

    // --- Write final solution to file ---
    FILE *fp = fopen("solution_bicgstab.txt", "w");
    fprintf(fp, "Total iterations: %d\n", iter);
    fprintf(fp, "Final solution vector x:\n");
    for (i = 0; i < N; i++)
        fprintf(fp, "%lf\n", x[i]);
    fclose(fp);

    clock_t toc = clock();
    printf("Elapsed: %lf seconds\n", (double)(toc - tic) / CLOCKS_PER_SEC);

    // --- Free allocated memory ---
    free(A); free(At); free(b); free(x); free(xn);
    free(r); free(rn); free(dum1); free(dum2);
    free(p); free(r2); free(s);

    return 0;
}


   
  
    
