#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<time.h>
#include"funcs.h"
#include<omp.h>
#define N 4
int Nthreads;


//==============================================================================
// Bi-conjugate Gradient Stabilized(BiCGSTAB) solver with OpenMP parallelization
//==============================================================================
int main(int argc, char **argv){

	// check numer of c.l. arguments
	if(argc < 2){
        printf("Usage: %s <Nthreads>\n", argv[0]);
        return -1;
    }
    // set number of OpenMP threads
    Nthreads=atoi(argv[1]);
    omp_set_num_threads(Nthreads);
    
    
    int i,j,flag=1,iter=0;
    double *A,*b,*x,*xn,*r,*rn,*dum1,*p,*dum2,*r2,*s;
    double e=0.000001,alpha,beta,w;
    
    // --- Allocate memory for required vectors and matrices ---     
    A =(double *)malloc(N*N * sizeof(double));
    b = (double *)malloc(N*sizeof(double));
    x = (double *)malloc(N*sizeof(double));
    xn = (double *)malloc(N*sizeof(double));
    r = (double *)malloc(N*sizeof(double));
    rn = (double *)malloc(N*sizeof(double));
    dum1 = (double *)malloc(N*sizeof(double));
    dum2 = (double *)malloc(N*sizeof(double));
    p = (double *)malloc(N*sizeof(double));
    r2 = (double *)malloc(N*sizeof(double));
    s = (double *)malloc(N*sizeof(double));


    // --- Read input matrix and RHS vector ---
    read_2d("mat.txt",&A[0],N);
    read_1d("vec.txt",b,N);
    
    //--- initial guess ---
    #pragma omp parallel for
    for(i=0;i<N;i++)
        x[i]=0;

    // --- initial residual r = b - A*x  (x=0 => r=b in our case)
    mul_mat_vec(&A[0],x,dum1,N);
    #pragma omp parallel for
    for(i=0;i<N;i++){   
        r[i] = b[i] - dum1[i];
        p[i]=r[i];
        r2[i] = r[i];  // r2 initialized with r1
    } 
    
    double t=omp_get_wtime(); // start times
    
    /*---------------------------------------------------------
      BiCGSTAB Iteration loop
    ---------------------------------------------------------*/
    while(flag==1){
        iter+=1;
        flag=0;
        
        // A*p -> dum1
        mul_mat_vec(&A[0],p,dum1,N);
        
         // alpha = (r.r2) / (r2.(A*p))
        alpha=mul_vec_vec(r,r2,N)/mul_vec_vec(r2,dum1,N);
        
         // s = r - alpha*A*p
        #pragma omp parallel for
        for(i=0;i<N;i++){
            s[i]=r[i]-alpha*dum1[i];
        }
        
        // A*s -> dum2
        mul_mat_vec(&A[0],s,dum2,N);
        
        // w = (A*s · s) / (A*s · A*s)
        w=mul_vec_vec(dum2,s,N)/mul_vec_vec(dum2,dum2,N);
        
        // xn = x + alpha*p + w*s
        // rn = s - w*A*s
        #pragma omp parallel for
        for(i=0;i<N;i++){   
            xn[i] = x[i]+alpha*p[i]+w*s[i];
            rn[i] = s[i]-w*dum2[i];
        } 
        
        // beta = ((rn·r2)/(r·r2)) * (alpha/w)
        beta=mul_vec_vec(rn,r2,N)/mul_vec_vec(r,r2,N) * (alpha/w);
        
        // p = rn + beta*(p - w*A*p)
        #pragma omp parallel for
        for(i=0;i<N;i++)
            p[i]=rn[i]+beta*(p[i]-w*dum1[i]);
            
        // convergence check
        #pragma omp parallel for reduction(|:flag)
        for(i=0;i<N;i++){
            if(fabs(x[i]-xn[i])>e)
                flag=1;
        }
        
        // update x and r
        #pragma omp parallel for
        for(i=0;i<N;i++){
	        x[i]=xn[i]; 
            r[i]=rn[i]; 
        }
    }
    /*---------------------------------------------------------
      Output and timing
    ---------------------------------------------------------*/
    printf("Total number of iterations %d\n",iter);
    printf("Elapsed: %lf seconds\n", omp_get_wtime()-t);

    /*---------------------------------------------------------
      Save final solution to file
    ---------------------------------------------------------*/
    FILE *fp = fopen("solution.txt","w");
    if(fp != NULL){
        for(i=0;i<N;i++)
            fprintf(fp,"%.8lf\n",x[i]);
        fclose(fp);
    } else {
        printf("Error: could not write solution file.\n");
    }

    /*---------------------------------------------------------
      Free memory
    ---------------------------------------------------------*/
    free(A); free(b); free(x); free(xn);
    free(r); free(rn); free(dum1); free(dum2);
    free(p); free(r2); free(s);

    return 0;
}


   
  
    
