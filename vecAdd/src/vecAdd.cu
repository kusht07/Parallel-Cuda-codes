//---------------------------------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
//---------------------------------------------------------------------------------
static const int WORK_SIZE = 200000000;
static const int BLK_SIZE = 256;

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
__global__ void vecAdd(unsigned int *A_d, unsigned int *B_d,
		unsigned int *C_d, int WORK_SIZE) {

	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	// **** Populate vecADD kernel function ****
	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	 // Get our global thread ID
	    int id = blockIdx.x*blockDim.x+threadIdx.x;

	    // Make sure we do not go out of bounds
	    if (id < WORK_SIZE)
	        C_d[id] = A_d[id] + B_d[id];


}



//---------------------------------------------------------------------------------
int main(void) {
	unsigned int *A_h;
	unsigned int *A_d;
	unsigned int *B_h;
	unsigned int *B_d;
	unsigned int *C_h;
	unsigned int *C_d;

	//Set Device
	CUDA_CHECK_RETURN(cudaSetDevice(0));

	//See random number generator
	srand(time(NULL));

	//Clear command prompt
	cout << "\033[2J\033[1;1H";

	cout << "Allocating arrays on host ... ";
	A_h = new unsigned int[WORK_SIZE];
	B_h = new unsigned int[WORK_SIZE];
	C_h = new unsigned int[WORK_SIZE];

	cout << "done.\nPopluating arrays on host ... ";
	for (int i = 0; i < WORK_SIZE; i++) {
		A_h[i] = rand();
		B_h[i] = rand();
	}

	cout << "done.\nAllocating arrays on device ... ";
	CUDA_CHECK_RETURN(
			cudaMalloc((void** ) &A_d, sizeof(unsigned int) * WORK_SIZE));
	CUDA_CHECK_RETURN(
			cudaMalloc((void** ) &B_d, sizeof(unsigned int) * WORK_SIZE));
	CUDA_CHECK_RETURN(
			cudaMalloc((void** ) &C_d, sizeof(unsigned int) * WORK_SIZE));

	cout << "done.\nCopying arrays from host to device ... ";
	CUDA_CHECK_RETURN(
			cudaMemcpy(A_d, A_h, sizeof(int) * WORK_SIZE,
					cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(
			cudaMemcpy(B_d, B_h, sizeof(int) * WORK_SIZE,
					cudaMemcpyHostToDevice));

	cout << "done.\nLaunching kernel ... \n";



    //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	// **** define kernel launch parameters ****
	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	 int threadsPerBlock,blocksPerGrid;


	  if (WORK_SIZE<BLK_SIZE){
	    threadsPerBlock = WORK_SIZE;
	    blocksPerGrid   = 1;
	  } else {
	    threadsPerBlock = BLK_SIZE;
	    blocksPerGrid   = ceil(double(WORK_SIZE)/double(threadsPerBlock));
	  }


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

	vecAdd<<< blocksPerGrid, threadsPerBlock >>>(A_d, B_d, C_d, WORK_SIZE);




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
			cudaMemcpy(C_h, C_d, sizeof(int) * WORK_SIZE,
					cudaMemcpyDeviceToHost));

	cout << "done.\nVerifying results on host ... ";

	//Add code to time host calculations
	clock_t st, ed;

	st = clock();

	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    // **** Check that results from kernel are correct ****
    // **** Complete validation code below             ****
	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

	bool valid = true;
	for (int i = 0; i < WORK_SIZE; i++) {
		if (C_h[i] != (A_h[i]+B_h[i]) ) {
			cout << "done.\n***GPU results are incorrect***";
			valid = false;
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
	CUDA_CHECK_RETURN(cudaFree((void* ) B_d));
	CUDA_CHECK_RETURN(cudaFree((void* ) C_d));
	CUDA_CHECK_RETURN(cudaDeviceReset());

	cout << "done.\nFreeing memory on host ... ";
	delete[] A_h;
	delete[] B_h;
	delete[] C_h;

	cout << "done.\nExiting program.\n";
	cout<<"  Kushagra Trivedi\n  3080669\n";
	return 0;
}
