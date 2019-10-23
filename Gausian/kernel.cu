
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


__device__ void printMatrix(float *matrix, unsigned dimension)
{
	printf("{\r\n");
	for (unsigned row = 0; row < dimension; row++)
	{
		printf("   {");
		for (unsigned column = 0; column < dimension; column++)
		{
			printf(" %.2f", matrix[column + row * dimension]);
		}

		printf("   }\r\n");
	}

	printf("}\r\n");
}


__device__ bool isCloseToZero(float value)
{
	if (value < 0)
	{
		value = -value;
	}

	return value < 0.000000000001;
}


__global__ void gaussianEliminationKernel(float* matrix, bool *isDegenerate, unsigned dimension, float *b, float* x)
{
	__shared__ unsigned swapWith;

	unsigned responsibleRow = blockDim.x * blockIdx.x + threadIdx.x;

	for (unsigned pivotRow = 0; pivotRow < dimension; pivotRow++)
	{
		if (responsibleRow == pivotRow)
		{
			float pivot;
			unsigned divisionRow = pivotRow-1;

			do
			{
				divisionRow++;
				pivot = matrix[divisionRow * dimension + pivotRow];
				printf("%d -- %.2f\r\n", divisionRow, pivot);
			} while (isCloseToZero(pivot) && divisionRow < dimension);

			swapWith = divisionRow;

			for (unsigned col = pivotRow; col < dimension; col++)
			{
				matrix[col + divisionRow * dimension] /= pivot;
			}

			b[divisionRow] /= pivot;

			printMatrix(matrix, dimension);
		}

		__syncthreads();

		if (swapWith != pivotRow)
		{
			printf("Swap %d with %d\r\n", pivotRow, swapWith);

			// Swapping phase, each thread is responsible for one column
			float temp = matrix[threadIdx.x + pivotRow * dimension];
			matrix[threadIdx.x + pivotRow * dimension] = matrix[threadIdx.x + swapWith * dimension];
			matrix[threadIdx.x + swapWith * dimension] = temp;

			// Thread 0 is responsible for b
			
			if (threadIdx.x == 0)
			{
				float temp = b[pivotRow];
				b[pivotRow] = b[swapWith];
				b[swapWith] = temp;
			}

			__syncthreads();
		
			if (threadIdx.x == 0)
			{
				printMatrix(matrix, dimension);
			}

			__syncthreads();
		}

		if (responsibleRow != pivotRow)
		{
			float leadingValue = matrix[pivotRow + responsibleRow * dimension];

			for (unsigned col = pivotRow; col < dimension; col++)
			{
				matrix[col + responsibleRow * dimension] -= leadingValue * matrix[col + pivotRow * dimension];
			}

			b[responsibleRow] -= b[pivotRow] * leadingValue;
		}

		__syncthreads();
	}

	if (threadIdx.x == 0)
	{
		printMatrix(matrix, dimension);
	}

	x[responsibleRow] = b[responsibleRow];
}

int main()
{
    const unsigned matrixSize = 3;
	float matrix[matrixSize * matrixSize] = {
		0, -4, 4,
		2, 3, -3,
		-2, -2, 1
	};

	float b[3] = { 5, -3, 2 };
	float x[3] = { 0, 0, 0 };

	float* deviceMatrix;
	float* deviceB;
	float* deviceX;

	cudaMalloc((void**)& deviceMatrix, sizeof(matrix));
	cudaMalloc((void**)& deviceB, sizeof(b));
	cudaMalloc((void**)& deviceX, sizeof(b));

	cudaMemcpy(deviceMatrix, matrix, sizeof(matrix), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceB, b, sizeof(b), cudaMemcpyHostToDevice);

	gaussianEliminationKernel<<<1, matrixSize>>>(deviceMatrix, matrixSize, deviceB, deviceX);

	cudaError_t cudaStatus = cudaDeviceSynchronize();

    // Add vectors in parallel.
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

	cudaMemcpy(x, deviceX, sizeof(x), cudaMemcpyDeviceToHost);
	
	printf("{ %.2f, %.2f, %.2f }\r\n", x[0], x[1], x[2]);

	cudaFree(deviceX);
	cudaFree(deviceMatrix);
	cudaFree(deviceB);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
