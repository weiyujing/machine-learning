//基于mkl的奇异值分解

#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include<algorithm>
#include "mkl.h"
using namespace std;
#define nn 5 //矩阵A的列数
#define nn2 3 //A的行数
typedef struct Tuple
{
	double r;
	double x[nn];
	
}Tuple;
typedef struct Tuple2
{
	double r;
	double x[nn2];

}Tuple2;
bool comparison(Tuple a, Tuple b) {
	return a.r>b.r;
}
bool comparison2(Tuple2 a, Tuple2 b) {
	return a.r>b.r;
}

int main()
{   
	Tuple t1[nn];//t1存储列方阵特征值
	Tuple2 t2[nn2];//t2存储行方阵特征值
	int matrix_order = LAPACK_COL_MAJOR;
	char jobvl = 'N';
	char jobvr = 'V';
	int ldaa = nn;
	double wr[nn] = { 0 };
	double wi[nn] = { 0 };
	double vl[nn*nn];
	int ldvl = nn;
	double vr[nn*nn];
	int ldvr = nn;

	int ldaa2 = nn2;
	double wr2[nn2] = { 0 };
	double wi2[nn2] = { 0 };
	double vl2[nn2*nn2];
	int ldvl2 = nn2;
	double vr2[nn2*nn2];
	int ldvr2 = nn2;
	
	/*
	参数M：表示 A或C的行数。如果A转置，则表示转置后的行数
	参数N：表示 B或C的列数。如果B转置，则表示转置后的列数。
	参数K：表示 A的列数或B的行数（A的列数 = B的行数）。如果A转置，则表示转置后的列数。
	参数LDA：表示A的列数，与转置与否无关。
	参数LDB：表示B的列数，与转置与否无关。
	参数LDC：始终 = N
	*/
	const int M = nn;    const int N = nn;    const int K = nn2;    
	const float alpha = 1;    const float beta = 0;    const int lda = nn;  
	const int ldb = nn;    const int ldc = N;    const float A[M*K] = { 0,1,1,1,1,0,2,1,3,0,5,1,2,3,4};       float C[M*N]; double C1[M*N]; 

	const int M2 = nn2;    const int N2 = nn2;    const int K2 = nn;
	  const int lda2 = nn;
	const int ldb2 = nn;    const int ldc2 = N2;      float C2[M2*N2];   double C3[M2*N2];
	double R[M*K] = {0};
	double U[nn2*nn2];//行

	double V[nn*nn];
	//AT*A
	cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, M, N, K, alpha, A, lda, A, ldb, beta, C, ldc);  //矩阵乘法    CblasTrans 和CblasNoTrans. 
	//A*AT
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M2, N2, K2, alpha, A, lda2, A, ldb2, beta, C2, ldc2);  //矩阵乘法    CblasTrans 和CblasNoTrans.
	for (int i = 0; i<M; i++)

	{
		for (int j = 0; j<N; j++)

		{
			C1[i*N + j] = (double)C[i*N + j];
		}
		
	}
	for (int i = 0; i<M2; i++)

	{
		for (int j = 0; j<N2; j++)

		{
			C3[i*N2 + j] = (double)C2[i*N2 + j];
		}
		
	}
	
	
	int info = LAPACKE_dgeev(matrix_order, jobvl, jobvr, nn, C1, ldaa, wr, wi, vl, ldvl, vr, ldvr);
	int info2 = LAPACKE_dgeev(matrix_order, jobvl, jobvr, nn2, C3, ldaa2, wr2, wi2, vl2, ldvl2, vr2, ldvr2);
	
	if (info == 0) {
		int i = 0;
		int j = 0;
	
		for (i = 0; i<nn; i++) {
			
			t1[i].r = wr[i];//存储特征值；
		
			for (j = 0; j < ldvr; j++) {
				
				t1[i].x[j] = vr[i*nn + j];//存储特征向量；
			}
		}
		sort(t1, t1 + nn, comparison);

	}

	if (info2 == 0) {
		int i = 0;
		int j = 0;
		int flag = 0;//区分复特征值的顺序
		for (i = 0; i<nn2; i++) {
			t2[i].r = wr2[i];
			
				for (j = 0; j<ldvr2; j++) {
					t2[i].x[j] = vr2[i*nn2 + j];		
				}
		}

		sort(t2, t2 + nn2, comparison2);
	}

	for (int j = 0; j <nn2; j++) {
		for (int i = 0; i < nn2; i++) {
			U[i*nn2 + j] = t2[j].x[i];
			
			
		}
	}
	for (int i = 0; i <nn2*nn2; i++) {
		if (i%nn2==0) {
			cout << endl;
		}
		cout << U[i] << "  ";	
	}
	cout << endl;

	for (int i = 0; i <nn; i++) {
		R[i*nn+i] = sqrt(t1[i].r);	
	}
	for (int i = 0; i <nn*nn2; i++) {
		if (i%nn == 0) {
			cout << endl;
		}

		cout << R[i] <<"  ";
	}
	
	cout << endl;


	for (int i = 0; i <nn; i++) {
		for (int j = 0; j < nn; j++) {
			V[i*nn + j] = t1[i].x[j];
		}
	}
	for (int i = 0; i <nn*nn; i++) {
		if (i%nn == 0) {
			cout << endl;
		}


		cout << V[i] <<"  ";
	}

	getchar();//必须要有这句

	return 0;
}
