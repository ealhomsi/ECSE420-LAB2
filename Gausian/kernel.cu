
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


__global__ void gaussianEliminationKernel(float* matrix, unsigned dimension, float *b, float* x, bool* isSingular)
{
	__shared__ int swapWith;

	unsigned responsibleRow = blockDim.x * blockIdx.x + threadIdx.x;

	for (unsigned pivotRow = 0; pivotRow < dimension; pivotRow++)
	{
		if (responsibleRow == pivotRow)
		{
			float pivot;
			swapWith = pivotRow-1;

			do
			{
				swapWith++;
				pivot = matrix[swapWith * dimension + pivotRow];
			} while (isCloseToZero(pivot) && swapWith < dimension);

			if (swapWith < dimension)
			{
				for (unsigned col = pivotRow; col < dimension; col++)
				{
					matrix[col + swapWith * dimension] /= pivot;
				}

				b[swapWith] /= pivot;
			}
			else
			{
				*isSingular = true;
			}

			printMatrix(matrix, dimension);
		}

		__syncthreads();

		if (swapWith >= dimension)
		{
			return;
		} 
		else if (swapWith != pivotRow)
		{
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
	bool* deviceSingular;

	cudaMalloc((void**)& deviceMatrix, sizeof(matrix));
	cudaMalloc((void**)& deviceB, sizeof(b));
	cudaMalloc((void**)& deviceX, sizeof(b));
	cudaMalloc((void**)& deviceSingular, sizeof(bool));

	cudaMemcpy(deviceMatrix, matrix, sizeof(matrix), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceB, b, sizeof(b), cudaMemcpyHostToDevice);

	gaussianEliminationKernel<<<1, matrixSize>>>(deviceMatrix, matrixSize, deviceB, deviceX, deviceSingular);

	bool singular;
	
	cudaError_t cudaStatus = cudaDeviceSynchronize();

    // Add vectors in parallel.
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

	cudaMemcpy(&singular, deviceSingular, sizeof(bool), cudaMemcpyDeviceToHost);
	cudaMemcpy(x, deviceX, sizeof(x), cudaMemcpyDeviceToHost);
	
	if (singular)
	{
		printf("The matrix is not invertible, there is no unique solution.");
	}
	else
	{
		printf("{ %.2f, %.2f, %.2f }\r\n", x[0], x[1], x[2]);
	}

	cudaFree(deviceSingular);
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
