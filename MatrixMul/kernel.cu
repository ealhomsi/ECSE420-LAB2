
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "A_1024.h"
#include "b_1024.h"
#include "X_1024.h"

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

__global__ void matrixMulKernel(double* deviceA, double* deviceX, double* deviceB, unsigned size, dim3 partitionSize)
{
	unsigned partitionX = blockIdx.x * blockDim.x + threadIdx.x;

	unsigned startX = partitionX * partitionSize.x, endX = (partitionX + 1) * partitionSize.x;

	for (unsigned i = startX; i < size && i < endX; i++)
	{
		deviceB[i] = 0.0;
		for (unsigned j = 0; j < size; j++)
		{
			deviceB[i] += deviceA[j + i * size] * deviceX[j];
		}
	}
}

unsigned getGpuProps(cudaDeviceProp* deviceProp)
{
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	if (deviceCount == 0) {
		printf("There is no device supporting CUDA\n");
		return 1;
	}

	cudaSetDevice(0);
	cudaGetDeviceProperties(deviceProp, 0);
	return 0;
}

double dabs(double x) {
	if (x < 0)
		return -x;
	return x;
}

long long getNanos()
{
	struct timespec ts;
	timespec_get(&ts, TIME_UTC);
	return (long long)ts.tv_sec * 1000000000L + ts.tv_nsec;
}

double relativeError(double computed, double real) {
	return dabs(computed - real) / real;
}

int main(int argc, char* argv[])
{
	if (argc != 3) {
		fprintf(stderr, "You should respect this format: ./MatrixMul < # threads> <# sample runs>");
		return 2;
	}
	unsigned numberOfThreads = atoi(argv[1]);
	unsigned sampleRuns = atoi(argv[2]);

	cudaDeviceProp deviceProps;
	unsigned errorCode = getGpuProps(&deviceProps);
	if (errorCode > 0)
	{
		return errorCode;
	}

	const unsigned matrixSize = std::sqrt(sizeof(A) / sizeof(double));

	double* deviceA;
	double* deviceB;
	double* deviceX;
	double* hostResultb = (double*)calloc(matrixSize, sizeof(double));


	cudaMalloc((void**)&deviceA, sizeof(A));
	cudaMalloc((void**)&deviceB, sizeof(b));
	cudaMalloc((void**)&deviceX, sizeof(X));

	cudaMemcpy(deviceA, A, sizeof(A), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceX, X, sizeof(X), cudaMemcpyHostToDevice);

	//matrixMulKernel << <1, matrixSize >> > (deviceMatrix, deviceX, matrixSize, , deviceX, );
	dim3 threads = dim3(fminf(numberOfThreads, matrixSize), 1, 1);
	unsigned xThreadsPerBlock = (unsigned)fminf(deviceProps.maxThreadsPerBlock, threads.x);

	dim3 threadLayout = dim3(xThreadsPerBlock);
	dim3 blockLayout = dim3((threads.x + xThreadsPerBlock - 1) / xThreadsPerBlock, 1, 1);
	dim3 partitionSize = dim3((matrixSize + threads.x - 1) / threads.x, 1, 1);


	long long startTime = getNanos();
	cudaError_t cudaStatus;
	for (int i = 0; i < sampleRuns; i++) {
		matrixMulKernel << <blockLayout, threadLayout >> > (deviceA, deviceX, deviceB, matrixSize, partitionSize);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, cudaGetErrorString(cudaStatus));
		}

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, cudaGetErrorString(cudaStatus));
		}
	}

	long long endTime = getNanos();
	long long averageNanos = (endTime - startTime) / sampleRuns;

	printf("Average ms using %d threads over %d trials: %.2f\r\n", numberOfThreads, sampleRuns, (double)averageNanos / 1000000);
	
	cudaStatus = cudaMemcpy(hostResultb, deviceB, sizeof(b), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, cudaGetErrorString(cudaStatus));
	}

	cudaFree(deviceA);
	cudaFree(deviceX);
	cudaFree(deviceB);

#ifdef NDEBUG
#else
	bool match = true;
	printf("\nb: {");
	for (int i = 0; i < matrixSize; i++) {
		double item = hostResultb[i];
		if (relativeError(item, b[i][0]) > 0.0001)  // this needs to account for numerical instablility
			match = false;
		printf("%.2f ", item);
	}
	printf("}\n");

	if (match) {
		printf("Performed A * X and got correct Matching on b\n");
	}
	else {
		printf("Performed A * X and got incorrect Matching on b\n");
	}
#endif

	free(hostResultb);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}
