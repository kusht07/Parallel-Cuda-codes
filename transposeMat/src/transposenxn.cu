//---------------------------------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>

//---------------------------------------------------------------------------------
static const int N = 10; //Number of rows and columns


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
__global__ void matrixTranspose(unsigned int* A_d, unsigned int *T_d, int n) {

	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	// **** Populate matrixTranspose kernel function ****
	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	 int i = threadIdx.x;
	 int j = threadIdx.y;

if(i<n&&j<n)
	     T_d[i+j*n] = A_d[j+i*n];

}


//---------------------------------------------------------------------------------
int main(void) {
	unsigned int **A ;
	unsigned int **T ;
	unsigned int *A_h;
	unsigned int *A_d;
	unsigned int *T_h;
	unsigned int *T_d;

	//Set Device
	CUDA_CHECK_RETURN(cudaSetDevice(0));

	//See random number generator
	srand(time(NULL));

	//Clear command prompt
	cout << "\033[2J\033[1;1H";

	cout << "Allocating arrays on host ... ";
	A_h = new unsigned int[N*N];
	T_h = new unsigned int[N*N];

	A = new unsigned int* [N];
	for (int i = 0; i < N; ++i) {
		A[i] = new unsigned int[N];
	}

	T = new unsigned int* [N];
	for (int i = 0; i < N; ++i) {
		T[i] = new unsigned int[N];
	}

	cout << "done.\nPopluating input matrix on host ...";
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			A[i][j] = rand();
		}
	}





	cout << "done.\nConverting 2-dimensional input matrix to 1-dimensional array on host ... ";


    //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	// **** Add code for converting 2-dimensional input matrix to 1-dimensional array here  ****
	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

	 for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {

			A_h[i*N+j] = A[i][j];

	}
	}


	cout << "done.\nAllocating arrays on device ... ";
	CUDA_CHECK_RETURN(
			cudaMalloc((void** ) &A_d, sizeof(unsigned int) * N*N));
	CUDA_CHECK_RETURN(
			cudaMalloc((void** ) &T_d, sizeof(unsigned int) * N*N));

	cout << "done.\nCopying arrays from host to device ... ";
	CUDA_CHECK_RETURN(
			cudaMemcpy(A_d, A_h, sizeof(int) * N*N,
					cudaMemcpyHostToDevice));

	cout << "done.\nLaunching kernel ... ";

    //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	// **** define kernel launch parameters ****
	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	dim3 dimGrid( 1,1, 1);
	dim3 dimBlock(10,10, 1);


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
	matrixTranspose<<< dimGrid, dimBlock >>>(A_d, T_d, N);


	CUDA_CHECK_RETURN(cudaEventRecord(stop, 0));

	CUDA_CHECK_RETURN(cudaEventSynchronize(stop));
	CUDA_CHECK_RETURN(cudaEventElapsedTime(&elapsedTime, start, stop));
	CUDA_CHECK_RETURN(cudaThreadSynchronize());	// Wait for the GPU launched work to complete
	CUDA_CHECK_RETURN(cudaGetLastError()); //Check if an error occurred in device code
	CUDA_CHECK_RETURN(cudaEventDestroy(start));
	CUDA_CHECK_RETURN(cudaEventDestroy(stop));
	cout << "done.\nElapsed kernel time: " << elapsedTime << " ms\n";

	cout << "Copying results back to host .... ";
	CUDA_CHECK_RETURN(
			cudaMemcpy(T_h, T_d, sizeof(int) * N*N,
					cudaMemcpyDeviceToHost));

	cout << "done.\nConverting 1-dimensional output array to 2-dimensional matrix on host ... ";

    //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	// **** Add code for converting 1-dimensional output array to 2-dimensional matrix here  ****
	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	for (int i = 0; i < N; ++i) {
			for (int j = 0; j < N; ++j) {

				T[i][j] =T_h[i*N+j] ;
			}

	}


	cout << "done.\nVerifying results on host ...";

	//Add code to time host calculations

	clock_t st, ed;

	st = clock();

	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    // **** Check that results from kernel are correct ****
    // **** Complete validation code below             ****
	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


	bool valid = true;
	for (int i = 0; i < N; ++i) {
			for (int j = 0; j < N; ++j) {
			if(T[i][j] != A[j][i])
			{
				cout << "\n\ndone.\n***GPU results are incorrect***";
				valid = false;
				break;
			}
		}
		if(!valid){
			break;
		}
	}

	cout << "done\n";

	if (valid) {
		cout << "GPU results are valid.\n";
	}


    	ed = clock() - st;
	cout << "Elapsed time on host: " << ((float) ed) / CLOCKS_PER_SEC * 1000
			<< " ms" << endl;

	cout << "Freeing memory on device ... ";
	CUDA_CHECK_RETURN(cudaFree((void* ) A_d));
	CUDA_CHECK_RETURN(cudaFree((void* ) T_d));
	CUDA_CHECK_RETURN(cudaDeviceReset());

	cout << "done.\nFreeing memory on host ... ";
	delete[] A_h;
	delete[] T_h;

	for (int i = 0; i < N; ++i) {
		delete[] A[i];
	}
	delete[] A;

	for (int i = 0; i < N; ++i) {
		delete[] T[i];
	}
	delete[] T;

	cout << "done.\nExiting program.\n";
	cout<<"  Kushagra Trivedi\n  3080669\n";
	return 0;
}

