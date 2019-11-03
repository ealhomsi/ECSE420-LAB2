
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "A.h"
#include "B.h"

#include <string.h>
#include <stdio.h>

#include "rational.h"
#include <math.h>
#include <float.h>
#include <stdio.h>



__device__ long gcd(long a, long b)
{
	long temp;

	while (b != 0)
	{
		temp = a % b;

		a = b;
		b = temp;
	}

	return a;
}


 long gcd2(long a, long b)
{
	long temp;

	while (b != 0)
	{
		temp = a % b;

		a = b;
		b = temp;
	}

	return a;
}

 double dabs(double x)
{
	return x > 0 ? x : -x;
}

 __device__ rational_t rational_scale(rational_t r, int scale, int inverse)
 {
	 if (inverse)
	 {
		 return  {
			 r.numerator / scale,
			 r.denominator / scale
		 };
	 }

	 return {
		 r.numerator * scale,
		 r.denominator * scale
	 };
 }

  rational_t rational_scale2(rational_t r, int scale, int inverse)
 {
	 if (inverse)
	 {
		 return  {
			 r.numerator / scale,
			 r.denominator / scale
		 };
	 }

	 return {
		 r.numerator * scale,
		 r.denominator * scale
	 };
 }

rational_t rational_reduced_form2(long numerator, long denominator)
{
	long scale = gcd2(numerator, denominator);
	rational_t r = {
		numerator,
		denominator
	};

	return rational_scale2(r, scale, 1);
}


__device__ rational_t rational_reduced_form(long numerator, long denominator)
{
	long scale = gcd(numerator, denominator);
	rational_t r = {
		numerator,
		denominator
	};

	return rational_scale(r, scale, 1);
}

rational_t rational_init(double x)
{
	long variable = 1;
	while (dabs(round(x) - x) > 2.2204460492503131e-016)
	{
		x *= 10;
		variable *= 10;
	}

	return rational_reduced_form2(
		(long)round(x),
		variable);
}

__device__ rational_t rational_negate(rational_t x)
{
	return {
		-x.numerator,
			x.denominator
	};
}

__device__ rational_t rational_add(rational_t x, rational_t y)
{
	if (y.denominator == x.denominator)
	{
		return rational_reduced_form(x.numerator + y.numerator, x.denominator);
	}

	rational_t xScaled = rational_scale(x, y.denominator, 0);
	rational_t yScaled = rational_scale(y, x.denominator, 0);

	return rational_reduced_form(
		xScaled.numerator + yScaled.numerator,
		xScaled.denominator);
}

__device__ rational_t rational_subtract(rational_t x, rational_t y)
{
	return rational_add(x, rational_negate(y));
}

__device__ rational_t rational_multiply(rational_t x, rational_t y)
{
	return rational_reduced_form(
		x.numerator * y.numerator,
		x.denominator * y.denominator);
}

__device__ rational_t rational_divide(rational_t x, rational_t y)
{
	// Assume denominator is always nonzero
	if (y.numerator == 0)
	{
		return RATIONAL_NAN;
	}

	return rational_reduced_form(
		x.numerator * y.denominator,
		x.denominator * y.numerator);
}

 double get_value(rational_t x) {
	return (1.0 * x.numerator) / x.denominator;
}

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

__device__ bool isCloseToZero(rational_t value)
{
	return value.numerator == 0;
}


__global__ void gaussianEliminationKernel(rational_t* matrix, unsigned dimension, rational_t* b, rational_t* x, bool* isSingular)
{
	__shared__ int swapWith;

	unsigned responsibleRow = blockDim.x * blockIdx.x + threadIdx.x;

	for (unsigned pivotRow = 0; pivotRow < dimension; pivotRow++)
	{
		if (responsibleRow == pivotRow)
		{
			rational_t pivot;
			swapWith = pivotRow - 1;

			do
			{
				swapWith++;
				pivot = matrix[swapWith * dimension + pivotRow];
			} while (isCloseToZero(pivot) && swapWith < dimension);

			if (swapWith < dimension)
			{
				for (unsigned col = pivotRow; col < dimension; col++)
				{
					matrix[col + swapWith * dimension] = rational_divide(matrix[col + swapWith * dimension], pivot);
				}

				b[swapWith] = rational_divide(b[swapWith], pivot);
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
			rational_t temp = matrix[threadIdx.x + pivotRow * dimension];
			matrix[threadIdx.x + pivotRow * dimension] = matrix[threadIdx.x + swapWith * dimension];
			matrix[threadIdx.x + swapWith * dimension] = temp;

			// Thread 0 is responsible for b

			if (threadIdx.x == 0)
			{
				rational_t temp = b[pivotRow];
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
			rational_t leadingValue = matrix[pivotRow + responsibleRow * dimension];

			for (unsigned col = pivotRow; col < dimension; col++)
			{
				matrix[col + responsibleRow * dimension] = rational_subtract(matrix[col + responsibleRow * dimension], rational_multiply(leadingValue, matrix[col + pivotRow * dimension]));
			}

			b[responsibleRow] = rational_subtract(b[responsibleRow], rational_multiply(leadingValue, b[pivotRow]));
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
	const unsigned matrixSize = (unsigned) sqrt(sizeof(A) / sizeof(double));

	rational_t* a = new rational_t[matrixSize * matrixSize];
	// convert double A to rational_t a
	for (int i = 0; i < matrixSize; i++) {
		for (int j = 0; j < matrixSize; j++) {
			a[j + i * matrixSize] = rational_init(A[i][j]);
		}
	}

	rational_t* b = new rational_t[matrixSize];
	// convert double B to rational_t b
	for (int i = 0; i < matrixSize; i++) {
		b[i] = rational_init(B[i][0]);
	}

	rational_t* x = new rational_t[matrixSize];
	// init rational_t x
	for (int i = 0; i < matrixSize; i++) {
		x[i] = rational_init(0.0);
	}

	rational_t* deviceMatrix;
	rational_t* deviceB;
	rational_t* deviceX;
	bool* deviceSingular;

	cudaMalloc((void**)& deviceMatrix, sizeof(rational_t) * matrixSize * matrixSize);
	cudaMalloc((void**)& deviceB, sizeof(rational_t) * matrixSize);
	cudaMalloc((void**)& deviceX, sizeof(rational_t) * matrixSize);
	cudaMalloc((void**)& deviceSingular, sizeof(bool));

	cudaMemcpy(deviceMatrix, a, sizeof(rational_t) * matrixSize * matrixSize, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceB, b, sizeof(rational_t) * matrixSize, cudaMemcpyHostToDevice);

	gaussianEliminationKernel << <1, matrixSize >> > (deviceMatrix, matrixSize, deviceB, deviceX, deviceSingular);

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
			rational_t item = x[i];
			printf("%.2f ", get_value(item));
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