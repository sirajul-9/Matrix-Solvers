#include<stdio.h>
#include<math.h>
#include<omp.h>
#include<stdlib.h>

#define N 4 //number of rows in matrix


//============================================================
// Function: jacobi()
//------------------------------------------------------------
// Solves Ax = b iteratively using the Jacobi method.
// Parallelized with OpenMP
//============================================================
void jacobi(double *a, double *b, double *x){
    double sum,err=0.000001;
    int i,j,iter=0,flag=1;
  
    // Allocate temporary array for new solution values
    double *x_new=(double *)calloc(N,sizeof(double));
  
    // ---Jacobi iteration loop ---
    while(flag==1){
        flag=0;
        iter+=1;
        
         // --- Compute new x values (parallelized over rows) ---
        #pragma omp parallel for private(j,sum) default(shared)
        for(i=0;i<N;i++){
            sum=0;
            for(j=0;j<N;j++){
                if(i!=j)
	                sum+=a[i*N+j]*x[j];
            }
            x_new[i]=(b[i]-sum)/a[i*N+i];
        }
        
        // --- Convergence check ---
        #pragma omp parallel for reduction(|:flag)
        for(i=0;i<N;i++){
            if(fabs(x[i]-x_new[i])>err)
                flag=1;  // any thread setting flag=1 keeps it set globally (bitwise OR reduction)
        }
        
         // --- Update x for next iteration ---
        #pragma omp parallel for
        for(i=0;i<N;i++)
	        x[i]=x_new[i];      
    }
    printf("Total number of iterations = %d\n",iter);
    free(x_new);
}


//============================================================
// main()
//------------------------------------------------------------
// Reads input matrix A and vector b from files "mat.txt" and "vec.txt".
// Initializes x = 0 and calls the parallel Jacobi solver.
//============================================================

int main(int argc, char **argv){
    double *A,*b,*x;
    float a;
    int i,j;
    
    // --- Allocate memory ---
    A =(double *)malloc(N*N * sizeof(double));
    x=(double *)malloc(N*sizeof(double));
    b=(double *)malloc(N*sizeof(double));
    
    // --- Set number of threads from command-line argument ---
    if (argc < 2) {
        printf("Usage: %s <num_threads>\n", argv[0]);
        return 1;
    }
    int Nthreads=atoi(argv[1]);
    omp_set_num_threads(Nthreads);
    printf("Running with %d OpenMP threads\n", Nthreads);
    
    // --- Read matrix and vector from file ---
    FILE *fp1 = fopen("mat.txt", "r");
    FILE *fp2 = fopen("vec.txt", "r");
    if (!fp1 || !fp2) {
        printf("Error: Unable to open input files.\n");
        return 1;
    }
    
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            if (fscanf(fp1, "%lf", &A[i*N+j]) != 1) {
                printf("Error reading data from the matrix file.\n");
                return 1;
            }
        }
    }  
    for(i=0;i<N;i++){
        if(fscanf(fp2,"%lf",&b[i]) !=1){
            printf("Error reading data from the vector file.\n");
                return 1;
        }
        x[i] = 0; //initialisation
    }
    
    fclose(fp1);
    fclose(fp2);

    // --- Run solver and measure wall time ---
    double start_time=omp_get_wtime();
    jacobi(A,b,x);
    double end_time=omp_get_wtime();
    
    
    // --- Print and save solution ---
	printf("Final solution vector x:\n");
	FILE *fp_out = fopen("sol.txt", "w");
	if (!fp_out) {
    	printf("Error: Unable to open sol.txt for writing.\n");
	} else {
    	for (i = 0; i < N; i++) {
    	    printf("%lf\n", x[i]);        // print to screen
    	    fprintf(fp_out, "%lf\n", x[i]); // save to file
    	}
    fclose(fp_out);
    printf("Solution written to sol.txt\n");
	}

	printf("Total time taken = %lf seconds\n", end_time - start_time);
    
    // ---free allocated memory---
    free(A);
    free(b);
    free(x);
    return 0;
}
