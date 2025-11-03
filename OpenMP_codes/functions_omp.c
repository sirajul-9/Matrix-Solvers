#include<stdio.h>
#include<stdlib.h>

extern int Nthreads;  // Declared in main solver file, specifies number of OpenMP threads


//============================================================
// Matrix–Vector Multiplication:  p = A * x
//============================================================
// Each row of A is multiplied with the vector x independently.
// Hence, this loop can be safely parallelized over 'i'.
//============================================================
void mul_mat_vec(double *A,double *x , double *p,long n){
    long i,j;
    double tmp;
    
     // Parallelize over rows
    #pragma omp parallel for private(j,tmp)
    for(i=0;i<n;i++){
        tmp=0;
        for(j=0;j<n;j++)
            tmp += *(A+i*n+j)*x[j];  // Dot product of row i of A with x
        p[i]=tmp;
    }
}


//============================================================
// Vector–Vector Dot Product:  c = aᵀ * b
//============================================================
// Each term a[i]*b[i] is independent; we use reduction to 
// safely accumulate the result across threads.
//============================================================
double mul_vec_vec(double *a, double *b,int n){
    int i;
    double c=0;
    
    #pragma omp parallel for private(i) reduction(+:c) default(shared)
    for(i=0;i<n;i++)
        c += a[i]*b[i];
        
    return c;
}


// Read a square matrix from file into array A (row-major order)
void read_2d(char *filename, double *A, int n){
    FILE *fp;
    fp=fopen(filename,"r");
    if(fp==NULL){
        printf("Error: Unable to open matrix file %s\n", filename);
        exit(1);
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (fscanf(fp, "%lf", (A+i*n+j)) != 1) {
                printf("Error reading matrix data from %s\n", filename);
                exit(2);
            }
        }
    } 
    fclose(fp); 
}


// Read a 1D vector from file into array b
void read_1d(char *filename, double *b,int n){
    FILE *fp;
    fp=fopen(filename,"r");
    if(fp==NULL){
        printf("Error: Unable to open vector file %s\n", filename);
        exit(3);
    }
    
    for(int i=0;i<n;i++){
        fscanf(fp,"%lf",&b[i]);
    }
    
    fclose(fp);
}


// Matrix Transpose:  At = Aᵀ
// Each element (i,j) of At = A[j,i]; independent per element,
// so we can parallelize over i.
void transpose(double *A,double *At, int n){
    int i,j;
    
    #pragma omp parallel for private(j) default(shared)
    for(i=0;i<n;i++)
        for(j=0;j<n;j++)
            *(At+i*n+j)= *(A+j*n+i);

}
