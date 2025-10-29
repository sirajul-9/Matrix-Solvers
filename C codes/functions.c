#include<stdio.h>
#include<stdlib.h>

//matrix vector multiplication
void mul_mat_vec(double *A,double *x , double *p,int n){
    int i,j;
    for(i=0;i<n;i++){
        p[i]=0;
        for(j=0;j<n;j++)
            p[i]+=*(A+i*n+j)*x[j];
    }
}

//vector vector multiplication
double mul_vec_vec(double *a, double *b,int n){
    int i;
    double c=0;
    for(i=0;i<n;i++)
        c+=a[i]*b[i];
    return c;
}

// function for reading the matrix A
void read_2d(char *filename, double *A, int n){
    FILE *fp;
    fp=fopen(filename,"r");
    if(fp==NULL){
        printf("unable to open file");
        exit(1);
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (fscanf(fp, "%lf", (A+i*n+j)) != 1) {
                printf("Error reading data from the file.\n");
                exit(2);
            }
        }
    } 
    fclose(fp); 
}


// function for rading the vector v
void read_1d(char *filename, double *b,int n){
    FILE *fp;
    fp=fopen(filename,"r");
    if(fp==NULL){
        printf("unable to open file");
        exit(3);
    }
    for(int i=0;i<n;i++){
        fscanf(fp,"%lf",&b[i]);
    }
    fclose(fp);
}

// function for matrix transpose
void transpose(double *A,double *At, int n){
    int i,j;
    for(i=0;i<n;i++)
        for(j=0;j<n;j++)
            *(At+i*n+j)= *(A+j*n+i);

}
