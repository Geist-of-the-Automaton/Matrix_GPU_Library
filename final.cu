#include "stdio.h"
const int threadsPerBlock = 1024;

//init ret as 0.0
__global__ void sum(float *data, float *ret, int *len) {
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ float c[threadsPerBlock];
	if (offset < *len)
		c[threadIdx.x] = data[offset];
	__syncthreads();
	for (int flag = 1; flag < blockDim.x; flag *= 2) {
		if (offset + flag < *len && threadIdx.x % (flag * 2) == 0)
			c[threadIdx.x] += c[threadIdx.x + flag];
		__syncthreads();
	}
	if (threadIdx.x == 0)
		atomicAdd(&ret, c[0]);
}

__global__ void clamp(float *data, float *min, float *max, int *len) {
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	if (offset < *len) {
		if (data[offset] < *min)
			data[offset] = *min;
		else if(data[offset] > *max)
			data[offset] = *max;
	}
}

__global__ void scalarAdd(float *data, float *val, int *len) {
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	if (offset < *len)
		data[offset] += *val;
}

__global__ void scalarSubtract(float *data, float *val, int *len) {
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	if (offset < *len)
		data[offset] -= *val;
}

__global__ void scalarMultiply(float *data, float *val, int *len) {
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	if (offset < *len)
		data[offset] *= *val;
}

__global__ void scalarDivide(float *data, float *val, int *len) {
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	if (offset < *len)
		data[offset] /= *val;
}

// do size comparison prior to call
__global__ void elementWiseAdd(float *data1, float *data2, int *len) {
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	if (offset < *len)
		data1[offset] += *data2[offset];
}

// do size comparison prior to call
__global__ void elementWiseSubtract(float *data1, float *data2, int *len) {
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	if (offset < *len)
		data1[offset] -= *data2[offset];
}

// do size comparison prior to call
__global__ void elementWiseMultiply(float *data1, float *data2, int *len) {
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	if (offset < *len)
		data1[offset] *= *data2[offset];
}

// do size comparison prior to call
__global__ void elementWiseDivide(float *data1, float *data2, int *len) {
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	if (offset < *len)
		data1[offset] /= *data2[offset];
}

//size comparison prior to call, must be linear. maybe
__global__ void reverse(float *data, int *len) {
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	if (offset < *len)
		data[offset] = -data[offset];
}

// linear matrix (vector) transpose will be cpu only. both will require realigning of offsets to 2d access
__global__ void transpose(float *data1, float *data2, int *width, int *height) {
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	if (offset < *width * *height) {
		int y = offset / *width, x = offset % *width;
		data2[x * height + y] = data1[offset]; //y * width + x
	}
}

//init ret as 0.0
__global__ void magnitude(float *data1, float *ret, int *len) {
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ float c[threadsPerBlock];
	if (offset < *len) {
		c[threadIdx.x] = data[offset];
		c[threadIdx.x] *= c[threadIdx.x];
	}
	__syncthreads();
	for (int flag = 1; flag < blockDim.x; flag *= 2) {
		if (offset + flag < *len && threadIdx.x % (flag * 2) == 0)
			c[threadIdx.x] += c[threadIdx.x + flag];
		__syncthreads();
	}
	if (threadIdx.x == 0)
		atomicAdd(&ret, c[0]);
}

//replaces difference in todo list
__global__ void absoluteValue(float *data1, float *data2, int *len) {
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	if (offset < *len) {
		if (data1[offset] < 0.0)
			data1[offset] = -data1[offset];
}

__global__ void vectorDotProduct(float *data1, float *data2, float *ret, int *len) {	
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ float c[threadsPerBlock];
	c[threadIdx.x] = data[offset] * B[offset];
	__syncthreads();
	for (int flag = 1; flag < blockDim.x; flag *= 2) {
		if (offset + flag < *len && threadIdx.x % (flag * 2) == 0)
			c[threadIdx.x] += c[threadIdx.x + flag];
		__syncthreads();
	}
	if (threadIdx.x == 0)
		atomicAdd(&ret, c[0]);
}

//check for square matrix
__global__ void trace(float *data, float *ret, int *width, int *height) {
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	if (offset < *width * *height) {
		int y = offset / *width, x = offset % *width;
		if (x == y)
			atomicAdd(&ret, data[offset]);
	}
}

// matrix dot product

// vector cross product

// matrix cross product

// normalize should make call to sum then scalar divide

//extract row

//extract col

// sum row

//sum rows

//sum col

//sum cols

struct matrix {
	int width = -1, height = -1;
	float **data;
	enum initType {identity, zeros, ones};

	matrix(int w, int h, initType type) {
		width = w;
		height = h;
		data = new float*[h];
		data[0] = new float[w * h];
		for (int i = 1; i < h; ++i)
			data[i] = data[i - 1] + w;
		for (int i = 0; i < w * h; ++i)
			data[0][i] = 0.0;
		switch (initType) {
			case identity:
				for (int i = 0; i < w * h; i += w + 1)
					size[0][i] = 1;
				break;
			case ones:
				for (int i = 0; i < w * h; ++i)
					size[0][i] = 0.0;
				break;
			case zeros:
				break;
		}
	}

	float sum() {

	}

	~matrix() {
		delete data;
	}
}

void GPU_big_dot(float *A, float *B, kernelType kt) {
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	int dataSize = vectSize * sizeof(float);
	float *dA, *dB, *dC, ms = 0.0;
	int *dLen;
	float *C = new float[dataSize];
	cudaMalloc(&dA, dataSize);
	cudaMalloc(&dB, dataSize);
	cudaMalloc(&dC, dataSize);
	cudaMalloc(&dLen, sizeof(int));
	cudaMemcpy(dA, A, dataSize, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, B, dataSize, cudaMemcpyHostToDevice);
	cudaMemcpy(dLen, &vectSize, sizeof(int), cudaMemcpyHostToDevice);
	int blocks = (vectSize + threadsPerBlock - 1) / threadsPerBlock;
	cudaEventRecord(start);
	if (kt == kernelType1)
		dotKernel1<<<blocks, threadsPerBlock>>>(dA, dB, dC, dLen);
	else
		dotKernel2<<<blocks, threadsPerBlock>>>(dA, dB, dC, dLen);
	cudaMemcpy(C, dC, dataSize, cudaMemcpyDeviceToHost);
	if (kt == kernelType1)
		for (int i = threadsPerBlock; i < vectSize; i += threadsPerBlock)
			C[0] += C[i];
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms, start, stop);
	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);
	printf(kt == kernelType1 ? "dot kernel 1 operated in %f milliseconds with result %f\n" : "dot kernel 2 operated in %f milliseconds with result %f\n", ms, C[0]);
}

int main( void ) {
	float *A = new float[vectSize], *B = new float[vectSize];
	for (int i = 0; i < vectSize; ++i) {
		A[i] = float(i);
		B[i] = float(i);
	}
	GPU_big_dot(A, B, kernelType1);
	GPU_big_dot(A, B, kernelType2);
	delete [] A;
	delete [] B;
 	return 0;
}

