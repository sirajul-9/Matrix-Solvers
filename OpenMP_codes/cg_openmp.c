#include<stdio.h>
#include<math.h>
#include<time.h>
#include<stdlib.h>
#include<omp.h>
#include"funcs.h"

#define N 4
int Nthreads;


//============================================================
// Conjugate Gradient (CG) solver with OpenMP parallelization
//============================================================
int main(int argc,char **argv){

	// set number of OpenMP threads from command line arguments 
    if (argc < 2) {
        printf("Usage: %s <num_threads>\n", argv[0]);
        return 1;
    }
    Nthreads = atoi(argv[1]);
    omp_set_num_threads(Nthreads);
    printf("Running with %d OpenMP threads\n", Nthreads);
    
    
    int i,j,flag=1,iter=0;
    double *A,*At,*b,*x,*xn,*r,*rn,*dum1,*p;
    double e=0.000001,alpha,beta;
    
    //---allocate memory---
    A =(double *)malloc(N*N * sizeof(double));
    At =(double *)malloc(N*N * sizeof(double));
    b = (double *)malloc(N*sizeof(double));
    x = (double *)calloc(N,sizeof(double)); //initilization with 0 done
    xn = (double *)malloc(N*sizeof(double));
    r = (double *)malloc(N*sizeof(double));
    rn = (double *)malloc(N*sizeof(double));
    dum1 = (double *)malloc(N*sizeof(double));
    p = (double *)malloc(N*sizeof(double));


    // Read input matrix/vector
    read_2d("mat.txt",A,N);
    read_1d("vec.txt",b,N);
    
    
    // ensure symmetry of A
    transpose(A,At,N);
    #pragma omp parallel for private(j)
    for(i=0;i<N;i++)
    	for(j=0;j<N;j++)
    		A[i*N+j]=(A[i*N+j]+At[i*N+j])/2;
    
    // Initial residual and direction
    mul_mat_vec(A,x,dum1,N);
    #pragma omp parallel for
    for(i=0;i<N;i++){   
        r[i] = b[i] - dum1[i];
        p[i]=r[i];
    } 
    
    double t_start = omp_get_wtime();

	// --- CG iteration loop ---
    while(flag==1){
        iter+=1;
        flag=0;
        
        mul_mat_vec(A,p,dum1,N);
        alpha=mul_vec_vec(r,r,N)/mul_vec_vec(p,dum1,N);
        
        #pragma omp parallel for
        for(i=0;i<N;i++){   
            xn[i] = x[i]+alpha*p[i];
            rn[i] = r[i]-alpha*dum1[i];
        } 
        
        beta=mul_vec_vec(rn,rn,N)/mul_vec_vec(r,r,N);
        
        #pragma omp parallel for
        for(i=0;i<N;i++)
            p[i]=rn[i]+beta*p[i];
            
        #pragma omp parallel for reduction(|:flag)
        for(i=0;i<N;i++){
            if(fabs(x[i]-xn[i])>e)
                flag=1;
        }
        
        #pragma omp parallel for
        for(i=0;i<N;i++){
	        x[i]=xn[i]; 
            r[i]=rn[i]; 
        }
    }
    double t_end = omp_get_wtime();

    // --- Output results ---
    printf("Total iterations: %d\n", iter);
    FILE *fp_out = fopen("sol.txt", "w");
    for (i = 0; i < N; i++) {
        printf("%lf\n", x[i]);
        fprintf(fp_out, "%lf\n", x[i]);
    }
    fclose(fp_out);
    printf("Solution written to sol.txt\n");
    printf("Elapsed time: %lf seconds\n", t_end - t_start);

    // --- Free memory ---
    free(A); free(At); free(b); free(x); free(xn);
    free(r); free(rn); free(p); free(dum1);

    return 0;
}
    
    
