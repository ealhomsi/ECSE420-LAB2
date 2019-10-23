#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "lodepng.h"
#include "wm.h"

#include <stdio.h>
#include <math.h>
#include <time.h>

union pixel {
	unsigned char channels[3];
	struct {
		unsigned char r;
		unsigned char g;
		unsigned char b;
	} colors;
};

__device__ unsigned char max(unsigned char a, unsigned char b)
{
	return a > b ? a : b;
}

__global__ void convolutionKernel(union pixel* inputImage, union pixel* outputImage, unsigned width, unsigned height, float *weights, unsigned kernelSize, dim3 partitionSize)
{
	unsigned inputWidth = width + kernelSize - 1;

	unsigned partitionX = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned partitionY = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned partitionZ = blockIdx.z;

	unsigned startX = partitionX * partitionSize.x, endX = (partitionX + 1) * partitionSize.x;
	unsigned startY = partitionY * partitionSize.y, endY = (partitionY + 1) * partitionSize.y;
	unsigned startZ = partitionZ * partitionSize.z, endZ = (partitionZ + 1) * partitionSize.z;

	for (unsigned pixelX = startX; pixelX < width && pixelX < endX; pixelX++)
	{
		for (unsigned pixelY = startY; pixelY < height && pixelY < endY; pixelY++)
		{
			for (unsigned channel = startZ; channel < 3 && channel < endZ; channel++)
			{
				unsigned char* output = &(outputImage[pixelX + width * pixelY].channels[channel]);

				float dot = 0.f;

				for (unsigned kX = 0; kX < kernelSize; kX++)
				{
					for (unsigned kY = 0; kY < kernelSize; kY++)
					{
						unsigned pixelValue = inputImage[pixelX + kX + inputWidth * (pixelY + kY)].channels[channel];
						float kernelValue = weights[kX + kernelSize * kY];

						dot += pixelValue * kernelValue;
					}
				}

				if (dot > 255)
				{
					*output = 255;
				}
				else if (dot < 0)
				{
					*output = 0;
				}
				else
				{
					*output = (unsigned char)dot;
				}
			}
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

dim3 partitionThreads(unsigned numberOfThreads, unsigned width, unsigned height)
{
	unsigned z = 1;

	if (numberOfThreads > width * height)
	{
		numberOfThreads /= 3;
		z = 3;
	}

	unsigned power = (unsigned)log2(numberOfThreads);
	return dim3(1 << (power / 2 + power % 2), 1 << (power / 2), z);
}

long long getNanos()
{
	struct timespec ts;
	timespec_get(&ts, TIME_UTC);
	return (long long)ts.tv_sec * 1000000000L + ts.tv_nsec;
}

int main(int argc, char* argv[])
{
	if (argc != 5) {
		fprintf(stderr, "You should respect this format: ./Convolution <name of input png> <name of output png> < # threads> <# Samples>");
		return 2;
	}

	char* inputFileName = argv[1];
	char* outputFileName = argv[2];
	unsigned numberOfThreads = atoi(argv[3]);
	unsigned numSamples = atoi(argv[4]);

	cudaDeviceProp deviceProps;
	unsigned errorCode = getGpuProps(&deviceProps);
	if (errorCode > 0)
	{
		return errorCode;
	}

	union pixel* imageData;
	unsigned inputWidth, inputHeight;
	unsigned kernelSize = (unsigned)sqrt(sizeof(w) / sizeof(float));

	unsigned returnCode = lodepng_decode24_file((unsigned char**)& imageData, &inputWidth, &inputHeight, inputFileName);

	unsigned outputWidth = inputWidth + 1 - kernelSize;
	unsigned outputHeight = inputHeight + 1 - kernelSize;

	if (returnCode != 0) {
		fprintf(stderr, "Error: reading the image file");
		return returnCode;
	}

	size_t pixelBufferSize = inputWidth * inputHeight * sizeof(union pixel);
	size_t outputBufferSize = outputWidth * outputHeight * sizeof(union pixel);
	union pixel* deviceInputImageData;
	union pixel* deviceOutputImageData;
	float* deviceWeights;

	union pixel* hostResultImageData = (union pixel*)calloc(outputWidth * outputHeight, sizeof(union pixel));

	cudaMalloc((void**)& deviceInputImageData, pixelBufferSize);
	cudaMalloc((void**)& deviceOutputImageData, outputBufferSize);
	cudaMalloc((void**)& deviceWeights, sizeof(w));
	cudaMemcpy(deviceInputImageData, imageData, pixelBufferSize, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceWeights, w, sizeof(w), cudaMemcpyHostToDevice);

	dim3 threads = partitionThreads(numberOfThreads, outputWidth, outputHeight);

	unsigned threadDimension = (unsigned)sqrtf(deviceProps.maxThreadsPerBlock);
	unsigned xThreadsPerBlock = (unsigned)fminf(threadDimension, threads.x);
	unsigned yThreadsPerBlock = (unsigned)fminf(threadDimension, threads.y);

	dim3 threadLayout = dim3(xThreadsPerBlock, yThreadsPerBlock);
	dim3 blockLayout = dim3((threads.x + xThreadsPerBlock - 1) / xThreadsPerBlock, (threads.y + yThreadsPerBlock - 1) / yThreadsPerBlock, threads.z);

	dim3 partitionSize = dim3((outputWidth + threads.x - 1) / threads.x, (outputHeight + threads.y - 1) / threads.y, 3 / threads.z);

	long long startTime = getNanos();

	for (int i = 0; i < numSamples; i++)
	{
		convolutionKernel<<<blockLayout, threadLayout>>>(
			deviceInputImageData, 
			deviceOutputImageData, 
			outputWidth, 
			outputHeight, 
			deviceWeights, 
			kernelSize, 
			partitionSize);

		cudaError_t cudaStatus = cudaGetLastError();
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
	long long averageNanos = (endTime - startTime) / numSamples;

	printf("Average ms over %d trials: %.2f\r\n", numSamples, (double)averageNanos / 1000000);

	cudaError_t cudaStatus = cudaMemcpy(hostResultImageData, deviceOutputImageData, outputBufferSize, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, cudaGetErrorString(cudaStatus));
	}

	cudaFree(deviceInputImageData);
	cudaFree(deviceOutputImageData);
	cudaFree(deviceWeights);

	// Save the image back to disk
	lodepng_encode24_file(outputFileName, (unsigned char*)hostResultImageData, outputWidth, outputHeight);

	free(hostResultImageData);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}



