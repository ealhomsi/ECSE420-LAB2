
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "A.h"
#include "b.h"
#include <string.h>

#include <stdio.h>


__device__ void printMatrix(double* matrix, double* b, unsigned dimension)
{
	printf("{\r\n");
	for (unsigned row = 0; row < dimension; row++)
	{
		printf("   {");
		for (unsigned column = 0; column < dimension; column++)
		{
			printf(" %.2f", matrix[column + row * dimension]);
		}
		printf("| %.2f", b[row]);
		printf(" }\r\n");
	}

	printf("}\r\n");
}

__device__ bool isCloseToZero(double value)
{
	if (value < 0)
	{
		value = -value;
	}

	return value < 0.0000000001;
}


__global__ void gaussianEliminationKernel(double* matrix, unsigned dimension, double *b, double* x, bool* isSingular)
{
	__shared__ int swapWith;

	unsigned responsibleRow = blockDim.x * blockIdx.x + threadIdx.x;

	for (unsigned pivotRow = 0; pivotRow < dimension; pivotRow++)
	{
		if (responsibleRow == pivotRow)
		{
			double pivot;
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

			//printMatrix(matrix, b, dimension);
		}

		__syncthreads();

		if (swapWith >= dimension)
		{
			return;
		} 
		else if (swapWith != pivotRow)
		{
			// Swapping phase, each thread is responsible for one column
			double temp = matrix[threadIdx.x + pivotRow * dimension];
			matrix[threadIdx.x + pivotRow * dimension] = matrix[threadIdx.x + swapWith * dimension];
			matrix[threadIdx.x + swapWith * dimension] = temp;

			// Thread 0 is responsible for b
			
			if (threadIdx.x == 0)
			{
				double temp = b[pivotRow];
				b[pivotRow] = b[swapWith];
				b[swapWith] = temp;
			}

			__syncthreads();
		
			/*if (threadIdx.x == 0)
			{
				printMatrix(matrix, b, dimension);
			}*/

			__syncthreads();
		}

		if (responsibleRow != pivotRow)
		{
			double leadingValue = matrix[pivotRow + responsibleRow * dimension];

			for (unsigned col = pivotRow; col < dimension; col++)
			{
				matrix[col + responsibleRow * dimension] -= leadingValue * matrix[col + pivotRow * dimension];
			}

			b[responsibleRow] -= b[pivotRow] * leadingValue;
		}

		__syncthreads();
	}

	/*if (threadIdx.x == 0)
	{
		printMatrix(matrix, b, dimension);
	}*/

	x[responsibleRow] = b[responsibleRow];
}

int main()
{
	const unsigned matrixSize = std::sqrt(sizeof(A) / sizeof(double));
	double * x = new double[matrixSize];

	double* deviceMatrix;
	double* deviceB;
	double* deviceX;
	bool* deviceSingular;

	cudaMalloc((void**)& deviceMatrix, sizeof(A));
	cudaMalloc((void**)& deviceB, sizeof(b));
	cudaMalloc((void**)& deviceX, sizeof(b));
	cudaMalloc((void**)& deviceSingular, sizeof(bool));

	cudaMemcpy(deviceMatrix, A, sizeof(A), cudaMemcpyHostToDevice);
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
	cudaMemcpy(x, deviceX, matrixSize * sizeof(double), cudaMemcpyDeviceToHost);
	
	if (singular)
	{
		// this hsould never happen
		printf("The matrix is not invertible, there is no unique solution.");
	}
	else
	{
		printf("\n{");
		for (int i = 0; i < matrixSize; i++) {
			double item = x[i];
			printf("%.2f ", item);
		}
		printf("}\n");

	}

	cudaFree(deviceSingular);
	cudaFree(deviceX);
	cudaFree(deviceMatrix);
	cudaFree(deviceB);
	delete[] x;

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
