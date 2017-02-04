//---------------------------------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>

//---------------------------------------------------------------------------------
	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	// **** 	A = M x N		****			AxB=C
     //****		B = N x K		****
  	 //**** 	C = M x K		****
	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


static const int M = 3;
static const int N = 5;
static const int K = 4;
static const int TILE_WIDTH = 2;

using namespace std;
//---------------------------------------------------------------------------------
/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }
//---------------------------------------------------------------------------------
__global__ void MatrixMulKernel(int ARows,int ACols, int BRows,
	    int BCols, int CRows, int CCols,unsigned int* A_d, unsigned int *B_d, unsigned int *C_d) {

	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	// **** Populate matrixMultiplication kernel function ****
	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


	int CValue = 0;

	    int Row = blockIdx.y*TILE_WIDTH + threadIdx.y;
	    int Col = blockIdx.x*TILE_WIDTH + threadIdx.x;

	    __shared__ int As[TILE_WIDTH][TILE_WIDTH];
	    __shared__ int Bs[TILE_WIDTH][TILE_WIDTH];

	    for (int k = 0; k < (TILE_WIDTH + ACols - 1)/TILE_WIDTH; k++) {

	         if (k*TILE_WIDTH + threadIdx.x < ACols && Row < ARows)
	             As[threadIdx.y][threadIdx.x] = A_d[Row*ACols + k*TILE_WIDTH + threadIdx.x];
	         else
	             As[threadIdx.y][threadIdx.x] = 0;

	         if (k*TILE_WIDTH + threadIdx.y < BRows && Col < BCols)
	             Bs[threadIdx.y][threadIdx.x] = B_d[(k*TILE_WIDTH + threadIdx.y)*BCols + Col];
	         else
	             Bs[threadIdx.y][threadIdx.x] = 0;

	         __syncthreads();

	         for (int n = 0; n < TILE_WIDTH; ++n)
	             CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];

	         __syncthreads();
	    }

	    if (Row < CRows && Col < CCols)
	        C_d[((blockIdx.y * blockDim.y + threadIdx.y)*CCols) +
	           (blockIdx.x * blockDim.x)+ threadIdx.x] = CValue;



	}




//---------------------------------------------------------------------------------
int main(void) {
	unsigned int **A ;
	unsigned int **B ;
	unsigned int **C ;
	unsigned int *A_h;
	unsigned int *A_d;
	unsigned int *B_h;
	unsigned int *B_d;
	unsigned int *C_h;
	unsigned int *C_d;
	unsigned int D[M][K];
	//Set Device
	CUDA_CHECK_RETURN(cudaSetDevice(0));

	//See random number generator
	srand(time(NULL));

	//Clear command prompt
	cout << "\033[2J\033[1;1H";

	cout << "Allocating arrays on host ... ";
	A_h = new unsigned int[M*N];
	B_h = new unsigned int[N*K];
	C_h = new unsigned int[M*K];

	A = new unsigned int* [M];
	for (int i = 0; i < M; ++i) {
		A[i] = new unsigned int[N];
	}

	B = new unsigned int* [N];
	for (int i = 0; i < N; ++i) {
		B[i] = new unsigned int[K];
	}
	C = new unsigned int* [M];
		for (int i = 0; i < M; ++i) {
			C[i] = new unsigned int[K];
		}
	cout << "done.\nPopluating input matrix on host ...";
	for (int i = 0; i < M; ++i) {
		for (int j = 0; j < N; ++j) {
			A[i][j] = rand()% 11;
		}
	}

	for (int i = 0; i < N; ++i) {
			for (int j = 0; j < K; ++j) {
				B[i][j] = rand()% 11;
			}
		}

	for (int i = 0; i < M; ++i) {
			for (int j = 0; j < K; ++j) {
				C[i][j] =0;
			}
		}
	cout << "done.\nConverting 2-dimensional input matrix to 1-dimensional array on host ... ";
	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
		// **** Add code for converting 2-dimensional input matrix to 1-dimensional array here  ****
		//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	for (int i = 0; i < M; ++i) {
		for (int j = 0; j < N; ++j) {
			A_h[i*N+j] = A[i][j];
		}
	}

	for (int i = 0; i < N; ++i) {
			for (int j = 0; j < K; ++j) {
				B_h[i*K+j] = B[i][j];
			}
		}
	cout << "done.\nAllocating arrays on device ... ";
	CUDA_CHECK_RETURN(
			cudaMalloc((void** ) &A_d, sizeof(unsigned int) * M*N));
	CUDA_CHECK_RETURN(
			cudaMalloc((void** ) &B_d, sizeof(unsigned int) * N*K));
	CUDA_CHECK_RETURN(
			cudaMalloc((void** ) &C_d, sizeof(unsigned int) * M*K));


	cout << "done.\nCopying arrays from host to device ... ";
	CUDA_CHECK_RETURN(
			cudaMemcpy(A_d, A_h, sizeof(int) * M*N,
					cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(
			cudaMemcpy(B_d, B_h, sizeof(int) * N*K,
						cudaMemcpyHostToDevice));

	cout << "done.\nLaunching kernel ... ";


    //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	// **** define kernel launch parameters ****
	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

	dim3 dimGrid(((K-1)/TILE_WIDTH+1), ((M-1)/TILE_WIDTH+1), 1);
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

	//Time kernel launch
	//Time kernel launch
	cudaEvent_t start, stop;
	CUDA_CHECK_RETURN(cudaEventCreate(&start));
	CUDA_CHECK_RETURN(cudaEventCreate(&stop));
	float elapsedTime;

	CUDA_CHECK_RETURN(cudaEventRecord(start, 0));



    //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	// **** Add kernel call here ****
	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	MatrixMulKernel<<< dimGrid, dimBlock >>>(M,N,N,K,M,K,A_d, B_d, C_d);


	CUDA_CHECK_RETURN(cudaEventRecord(stop, 0));

	CUDA_CHECK_RETURN(cudaEventSynchronize(stop));
	CUDA_CHECK_RETURN(cudaEventElapsedTime(&elapsedTime, start, stop));
	CUDA_CHECK_RETURN(cudaThreadSynchronize());	// Wait for the GPU launched work to complete
	CUDA_CHECK_RETURN(cudaGetLastError()); //Check if an error occurred in device code
	CUDA_CHECK_RETURN(cudaEventDestroy(start));
	CUDA_CHECK_RETURN(cudaEventDestroy(stop));
	cout << "done.\nElapsed kernel time: " << elapsedTime << " ms\n";

	cout << "Copying results back to host .... \n";
	CUDA_CHECK_RETURN(
			cudaMemcpy(C_h, C_d, sizeof(int) * M*K,
					cudaMemcpyDeviceToHost));


	cout << "done.\nConverting 1-dimensional output array to 2-dimensional matrix on host ... ";

    //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	// **** Add code for converting 1-dimensional output array to 2-dimensional matrix here  ****
	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	for (int i = 0; i < M; ++i) {
			for (int j = 0; j < K; ++j) {

				C[i][j] =C_h[i*K+j] ;
			}

	}




	clock_t st, ed;

	st = clock();



	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	    // **** Check that results from kernel are correct ****
	    // **** Complete validation code below             ****
		//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@



	 for(int i=0;i<M;++i)
	        {
	            for(int j=0;j<K;++j)
	            {
	                D[i][j]=0;
	                for(int k=0;k<N;++k)
	                    D[i][j]=D[i][j]+(A[i][k]*B[k][j]);

	        }

	        }

		bool valid = true;
		for (int i = 0; i < M; ++i) {
				for (int j = 0; j < K; ++j) {
				if(C[i][j] != D[i][j])
				{
					cout << "\ndone.\n***GPU results are incorrect***";
					valid = false;
					break;
				}
			}
			if(!valid){
				break;
			}
		}

	cout<<"done\n";
	if (valid) {
			cout << "GPU results are valid.\n";
	}

    	ed = clock() - st;
	cout << "Elapsed time on host: " << ((float) ed) / CLOCKS_PER_SEC * 1000
			<< " ms" << endl;

	cout << "Freeing memory on device ... ";
	CUDA_CHECK_RETURN(cudaFree((void* ) A_d));
	CUDA_CHECK_RETURN(cudaFree((void* ) B_d));
	CUDA_CHECK_RETURN(cudaFree((void* ) C_d));
	CUDA_CHECK_RETURN(cudaDeviceReset());

	cout << "done.\nFreeing memory on host ... ";
	delete[] A_h;
	delete[] B_h;
	delete[] C_h;

	for (int i = 0; i < M; ++i) {
		delete[] A[i];
	}
	delete[] A;

	for (int i = 0; i < N; ++i) {
		delete[] B[i];
	}
	delete[] B;

	cout << "done.\nExiting program.\n";
	cout<<"  Kushagra Trivedi\n  3080669\n";
	return 0;
}

