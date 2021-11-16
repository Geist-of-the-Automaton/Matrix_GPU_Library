#include "stdio.h"
const int threadsPerBlock = 1024;

// we need to examine the maximum input vector/matrix size allowed by each by the limitations of the algoritm implemented.

//init ret as 0.0
__global__ void sumGPU(float *data, float *ret, int *len) {
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

void sumCPU(float *data, float *ret, int *len) {
	float val = 0.0;
	int length = *len;
	for (int i = 0; i < length; ++i)
		val += data[i];
	*ret = val;
}
 
__global__ void clampGPU(float *data, float *min, float *max, int *len) {
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	if (offset < *len) {
		if (data[offset] < *min)
			data[offset] = *min;
		else if(data[offset] > *max)
			data[offset] = *max;
	}
}

void clampCPU(float *data, float *min, float *max, int *len) {
	int length = *len;
	float Min = *min, Max = *max;
	for (int i = 0; i < length; ++i) {
		if (data[i] < Min)
			data[i] = Min;
		else if(data[i] > Max)
			data[i] = Max;
	}
}

__global__ void scalarAddGPU(float *data, float *val, int *len) {
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	if (offset < *len)
		data[offset] += *val;
}

void scalarAddCPU(float *data, float *val, int *len) {
	float value = *val;
	int length = *len;
	for (int i = 0; i < length; ++i)
		data[i] += value;
}

__global__ void scalarSubtractGPU(float *data, float *val, int *len) {
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	if (offset < *len)
		data[offset] -= *val;
}

void scalarSubtractCPU(float *data, float *val, int *len) {
	float value = *val;
	int length = *len;
	for (int i = 0; i < length; ++i)
		data[i] -= value;
}

__global__ void scalarMultiplyGPU(float *data, float *val, int *len) {
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	if (offset < *len)
		data[offset] *= *val;
}

void scalarMultiplyCPU(float *data, float *val, int *len) {
	float value = *val;
	int length = *len;
	for (int i = 0; i < length; ++i)
		data[i] *= value;
}

__global__ void scalarDivideGPU(float *data, float *val, int *len) {
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	if (offset < *len)
		data[offset] /= *val;
}

void scalarDivideCPU(float *data, float *val, int *len) {
	float value = *val;
	int length = *len;
	for (int i = 0; i < length; ++i)
		data[i] /= value;
}

// do size comparison prior to call
__global__ void elementWiseAddGPU(float *data1, float *data2, int *len) {
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	if (offset < *len)
		data1[offset] += data2[offset];
}

void elementWiseAddCPU(float *data1, float *data2, int *len) {
	int length = *len;
	for (int i = 0; i < length; ++i)
		data1[i] += data2[i];
}

// do size comparison prior to call
__global__ void elementWiseSubtractGPU(float *data1, float *data2, int *len) {
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	if (offset < *len)
		data1[offset] -= data2[offset];
}

void elementWiseSubtractCPU(float *data1, float *data2, int *len) {
	int length = *len;
	for (int i = 0; i < length; ++i)
		data1[i] -= data2[i];
}

// do size comparison prior to call
__global__ void elementWiseMultiplyGPU(float *data1, float *data2, int *len) {
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	if (offset < *len)
		data1[offset] *= data2[offset];
}

void elementWiseMultiplyCPU(float *data1, float *data2, int *len) {
	int length = *len;
	for (int i = 0; i < length; ++i)
		data1[i] *= data2[i];
}

// do size comparison prior to call
__global__ void elementWiseDivideGPU(float *data1, float *data2, int *len) {
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	if (offset < *len)
		data1[offset] /= data2[offset];
}

void elementWiseDivideCPU(float *data1, float *data2, int *len) {
	int length = *len;
	for (int i = 0; i < length; ++i)
		data1[i] += data2[i];
}

//size comparison prior to call, must be linear. maybe
__global__ void reverseGPU(float *data, int *len) {
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	if (offset < *len)
		data[offset] = -data[offset];
}

void reverseCPU(float *data, int *len) {
	int length = *len;
	for (int i = 0; i < length; ++i)
		data[i] = -data[i];
}

// linear matrix (vector) transpose will be cpu only. both will require realigning of offsets to 2d access
__global__ void transposeGPU(float *data1, float *retData, int *width, int *height) {
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	if (offset < *width * *height) {
		int y = offset / *width, x = offset % *width;
		retData[x * *height + y] = data1[offset]; //y * width + x
	}
}

void transposeCPU(float *data1, float *retData, int *width, int *height) {
	int w = *width, h = *height;
	for (int i = 0; i < w * h; ++i) {
		int y = offset / w, x = offset % w;
		retData[x * h + y] = data1[offset];
	}
}

//init ret as 0.0, sqrt upon return
__global__ void magnitudeGPU(float *data1, float *ret, int *len, int *lock) {
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
	if (threadIdx.x == 0)
		atomicAdd(lock, 1);
	__syncthreads();
	if (threadIdx.x == 0) {
		atomicAdd(lock, -1);
		while (*lock != 0);
	}
	__syncthreads();
	*ret = sqrt(*ret);
}

//sqrt upon return
void magnitudeCPU(float *data1, float *ret, int *len, int *lock) {
	float val = 0.0;
	int length = *len;
	for (int i = 0; i < length; ++i)
		val += data[i] * data[i];
	*ret = sqrt(val);
}

//replaces absolute value in todo list when data2 is 0 matrix
__global__ void differenceGPU(float *data1, float *data2, int *len) {
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	if (offset < *len)
		data1[offset] = abs(data1[offset] - data2[offset]);
}

void differenceCPU(float *data1, float *data2, int *len) {
	int length = *len;
	for (int i = 0; i < length; ++i)
		data1[i] = abs(data1[i] - data2[offset]);
}

__global__ void vectorDotProductGPU(float *data1, float *data2, float *ret, int *len) {	
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ float c[threadsPerBlock];
	c[threadIdx.x] = data1[offset] * data2[offset];
	__syncthreads();
	for (int flag = 1; flag < blockDim.x; flag *= 2) {
		if (offset + flag < *len && threadIdx.x % (flag * 2) == 0)
			c[threadIdx.x] += c[threadIdx.x + flag];
		__syncthreads();
	}
	if (threadIdx.x == 0)
		atomicAdd(&ret, c[0]);
}

void vectorDotProductCPU(float *data1, float *data2, float *ret, int *len) {	
	float val = 0.0;
	int length = *len;
	for (int i = 0; i < length; ++i)
		val += data1[i] * data2[i];
	*ret = val;
}

__global__ void matrixDotProductGPU(float *data1, float *data2, float *retData, int *w2, int *common, int *len) {	
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ int W = *w2, C = *common;
	if (offset < *len) {
		retData[offset] = 0.0;
		int i = offset % W;
		int j = offset / W;
		for (int k = 0; k < C; ++k)
			retData[offset] += data1[j * C + k] * data2[k * W + i];
	}
}

void matrixDotProductCPU(float *data1, float *data2, float *retData, int *w2, int *common, int *len) {
	int length = *len, int W = *w2, C = *common;
	for (int i = 0; i < length; ++i) {
		retData[i] = 0.0;
		int i = offset % W;
		int j = offset / W;
		for (int k = 0; k < C; ++k)
			retData[offset] += data1[j * C + k] * data2[k * W + i];
	}
}

//check for square matrix
__global__ void traceGPU(float *data, float *ret, int *width, int *height) {
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	if (offset < *width * *height) {
		int y = offset / *width, x = offset % *width;
		if (x == y)
			atomicAdd(&ret, data[offset]);
	}
}

void traceCPU(float *data, float *ret, int *width, int *height) {
	int w = *width, h = *height;
	float val = 0.0;
	for (int i = 0; i < w * h; i += w + 1)
		val += data[i];
	*ret = val;
}

//shape check prior to call in struct
void vectorCrossProduct(float *data1, float *data2, float *retData) {
	retData[0] = data1[1] * data2[2] - data1[2] * data2[1];
	retData[1] = data1[2] * data2[0] - data1[0] * data2[2];
	retData[2] = data1[0] * data2[1] - data1[1] * data2[0];
}

//row and bounds check prior to call in struct
void extractRow(float *data, float *retData, int row, int width) {
	memcpy(retData, data[row * width], width * sizeof(float));
}

//row and bounds check prior to call in struct
void extractCol(float *data, float *retData, int col, int width, int height) {
	for (int i = 0; i < height; ++i)
		retData[i] = data[i * width];
}

// UNLESS THIS IS GENERALIZED FOR LEFT AND RIGHT INVERSES ON RECTANGULAR MATRICIES, WIDTH AND
// HEIGHT CAN BE REPLACED WITH SIZE
//inverse needs to be fed an identity matrix
__global__ void inverseGPU(float *data, float *inverse, int *width, int *height, int *lock) {
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ int W = *width; 
	for (int row = 0; row < *height; ++row) {
		int H = row * W;
		if (data[H + row] == 0.0) {
			if (offset == 0)
				printf("Error: Zero found on leading diagonal of matrix at %d, %d", row, row);
			break;
		}
		if (threadIdx.x == 0)
			atomicAdd(lock, 1);
		if (offset > row)
			data[H + offset] /= data[H + row];
		inverse[H + offset] /= data[H + row];
		__syncthreads();
		if (threadIdx.x == 0) {
			atomicAdd(lock, -1);
			while (*lock != 0);
		}
		__syncthreads();
		if (offset == 0)
			data[H + row] = 1.0;
		if (offset > row) {
			for (int r = row + 1; r < *height; ++r)
				data[r * W + offset] -= data[H + offset] * data[r * W + row];
		}
		for (int r = row + 1; r < *height; ++r)
			inverse[r * W + offset] -= inverse[H + offset] * data[r * W + row];
		__syncthreads();
		if (threadIdx.x == 0) {
			atomicAdd(lock, -1);
			while (*lock != 0);
		}
		__syncthreads();
		if (offset > row)
			data1[offset * W + row] = 0.0;
	}
	for (int row = 1; row < *height; ++row) {
		int H = row * W;
		if (threadIdx.x == 0)
			atomicAdd(lock, 1);
		if (offset > row) {
			for (int r = row - 1; r >= 0; --r)
				data[r * W + offset] -= data[H + offset] * data[r * W + row];
		}
		for (int r = row - 1; r >= 0; --r)
			inverse[r * W + offset] -= inverse[H + offset] * data[r * W + row];
		__syncthreads();
		if (threadIdx.x == 0) {
			atomicAdd(lock, -1);
			while (*lock != 0);
		}
		__syncthreads();
		if (offset < row)
			data[offset * W + row] = 0.0;
	}
}

// SEE COMMENT FOR INVERSE ABOVE
//inverse needs to be fed an identity matrix
__global__ void pInverseGPU(float *data, float *inverse, float *identity, int *width, int *height, int *lock) {
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ int W = *width; 
	float val = 0.0;
	int index = 0;
	if (threadIdx.x == 0)
		atomicAdd(lock, 1);
	__syncthreads();
	if (threadIdx.x == 0) {
		atomicAdd(lock, -1);
		while (*lock != 0);
	}
	__syncthreads();
	for (int row = 0; row < *height; ++row) {
		int H = row * W;
		if (threadIdx.x == 0)
			atomicAdd(lock, 1);
		__syncthreads();
		if (threadIdx.x == 0) {
			if (offset == 0) {
				val = 0.0;
				index = row;
				for (int r = row + 1; r < *height; ++r) {
					float abs = data[r * W + row];
					if (abs < 0.0)
						abs = -abs;
					if (data[r * W + row] >= val)
						index = r;
				}
				if (index != row) {
					val = data[H + row];
					data[H + row] = data[index * W + row];
					data[index * W + row] = val;
				}
			}
			atomicAdd(lock, -1);
			while (*lock != 0);
		}
		__syncthreads();
		if (data[H + row] == 0.0) {
			if (offset == 0)
				printf("Error: Zero found on leading diagonal of matrix at %d, %d", row, row);
			break;
		}
		if (index != row) {
			val = identity[H + offset];
			identity[H + offset] = identity[index * W + offset];
			identity[index * W + offset] = val;
			if (offset > row + 1 && offset < W) {
				val = data[H + offset];
				data[H + offset] = data[index * W + offset];
				data[index * W + offset] = val;
			}
		}
		if (threadIdx.x == 0)
			atomicAdd(lock, 1);
		if (offset > row)
			data[H + offset] /= data[H + row];
		inverse[H + offset] /= data[H + row];
		__syncthreads();
		if (threadIdx.x == 0) {
			atomicAdd(lock, -1);
			while (*lock != 0);
		}
		__syncthreads();
		if (offset == 0)
			data[H + row] = 1.0;
		if (offset > row) {
			for (int r = row + 1; r < *height; ++r)
				data[r * W + offset] -= data[H + offset] * data[r * W + row];
		}
		for (int r = row + 1; r < *height; ++r)
			inverse[r * W + offset] -= inverse[H + offset] * data[r * W + row];
		__syncthreads();
		if (threadIdx.x == 0) {
			atomicAdd(lock, -1);
			while (*lock != 0);
		}
		__syncthreads();
		if (offset > row)
			data[offset * W + row] = 0.0;
	}
	for (int row = 1; row < *height; ++row) {
		int H = row * W;
		if (threadIdx.x == 0)
			atomicAdd(lock, 1);
		if (offset > row) {
			for (int r = row - 1; r >= 0; --r)
				data[r * W + offset] -= data[H + offset] * data[r * W + row];
		}
		for (int r = row - 1; r >= 0; --r)
			inverse[r * W + offset] -= inverse[H + offset] * data[r * W + row];
		__syncthreads();
		if (threadIdx.x == 0) {
			atomicAdd(lock, -1);
			while (*lock != 0);
		}
		__syncthreads();
		if (offset < row)
			data[offset * W + row] = 0.0;
	}
}

// GENERALIZE FOR NON SQUARES
// data becomes upper, lower mustbe fed an identity matrix
__global__ void luDecompSquareGPU(float *data, float *lower, int *width, int *height, int *lock) {
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ int W = *width; 
	for (int row = 0; row < *height; ++row) {
		int H = row * W;
		if (data[H + row] == 0.0) {
			if (offset == 0)
				printf("Error: Zero found on leading diagonal of matrix at %d, %d", row, row);
			break;
		}
		if (threadIdx.x == 0)
			atomicAdd(lock, 1);
		if (offset > row) {
			lower[offset * W + row] = data[offset * W + row] / data[H + row];
			for (int r = row + 1; r < *height; ++r)
				data[r * W + offset] -= data[H + offset] * data[r * W + row] / data[H + row];
		}
		__syncthreads();
		if (threadIdx.x == 0) {
			atomicAdd(lock, -1);
			while (*lock != 0);
		}
		__syncthreads();
	}
}

// GENERALIZE FOR NON SQUARES
__global__ void pluDecompSquareGPU(float *data, float *lower, float *identity, int *size, int *lock) {
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ int W = *width; 
	float val = 0.0;
	int index = 0;
	if (threadIdx.x == 0)
		atomicAdd(lock, 1);
	__syncthreads();
	if (threadIdx.x == 0) {
		atomicAdd(lock, -1);
		while (*lock != 0);
	}
	__syncthreads();
	for (int row = 0; row < *height; ++row) {
		int H = row * W;
		if (threadIdx.x == 0)
			atomicAdd(lock, 1);
		__syncthreads();
		if (threadIdx.x == 0) {
			if (offset == 0) {
				val = abs(data[r * W + row]);
				index = row;
				for (int r = row + 1; r < *height; ++r) {
					float abs = abs(data[r * W + row]);
					if (abs >= val) {
						index = r;
						val = abs;
					}
				}
				if (index != row) {
					val = data[H + row];
					data[H + row] = data[index * W + row];
					data[index * W + row] = val;
				}
			}
			atomicAdd(lock, -1);
			while (*lock != 0);
		}
		__syncthreads();
		if (data[H + row] == 0.0) {
			if (offset == 0)
				printf("Error: Zero found on leading diagonal of matrix at %d, %d; permutation unsuccessful", row, row);
			break;
		}
		if (index != row) {
			val = identity[H + offset];
			identity[H + offset] = identity[index * W + offset];
			identity[index * W + offset] = val;
			if (offset > row + 1 && offset < W) {
				val = data1[H + offset];
				data1[H + offset] = data1[index * W + offset];
				data1[index * W + offset] = val;
			}
		}
		if (threadIdx.x == 0)
			atomicAdd(lock, 1);
		if (offset > row) {
			lower[offset * W + row] = data[offset * W + row] / data[H + row];
			for (int r = row + 1; r < *height; ++r)
				data1[r * W + offset] -= data1[H + offset] * data1[r * W + row] / data1[H + row];
		}
		__syncthreads();
		if (threadIdx.x == 0) {
			atomicAdd(lock, -1);
			while (*lock != 0);
		}
		__syncthreads();
	}
}

//inverse needs to be fed an identity matrix
// GENERALIZE FOR NON SQUARE
__global__ void solveLinearSysGPU(float *data, float *equs, float *identity, int *width, int *height, int *lock) {
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ int W = *width; 
	float val = 0.0;
	int index = 0;
	if (threadIdx.x == 0)
		atomicAdd(lock, 1);
	__syncthreads();
	if (threadIdx.x == 0) {
		atomicAdd(lock, -1);
		while (*lock != 0);
	}
	__syncthreads();
	for (int row = 0; row < *height; ++row) {
		int H = row * W;
		if (threadIdx.x == 0)
			atomicAdd(lock, 1);
		__syncthreads();
		if (threadIdx.x == 0) {
			if (offset == 0) {
				val = abs(data[r * W + row]);
				index = row;
				for (int r = row + 1; r < *height; ++r) {
					float abs = abs(data[r * W + row]);
					if (abs >= val) {
						index = r;
						val = abs;
					}
				}
				if (index != row) {
					val = data[H + row];
					data[H + row] = data[index * W + row];
					data[index * W + row] = val;
				}
			}
			atomicAdd(lock, -1);
			while (*lock != 0);
		}
		__syncthreads();
		if (data[H + row] == 0.0) {
			if (offset == 0)
				printf("Error: Zero found on leading diagonal of matrix at %d, %d", row, row);
			break;
		}
		if (index != row) {
			val = identity[H + offset];
			identity[H + offset] = identity[index * W + offset];
			identity[index * W + offset] = val;
			if (offset > row + 1 && offset < W) {
				val = data[H + offset];
				data[H + offset] = data[index * W + offset];
				data[index * W + offset] = val;
			}
		}
		if (threadIdx.x == 0)
			atomicAdd(lock, 1);
		if (offset > row)
			data[H + offset] /= data[H + row];
		__syncthreads();
		if (threadIdx.x == 0) {
			atomicAdd(lock, -1);
			while (*lock != 0);
		}
		__syncthreads();
		if (offset == 0)
			data[H + row] = 1.0;
		if (offset > row)
			for (int r = row + 1; r < *height; ++r)
				data[r * W + offset] -= data[H + offset] * data[r * W + row];
		__syncthreads();
		if (threadIdx.x == 0) {
			atomicAdd(lock, -1);
			while (*lock != 0);
		}
		__syncthreads();
		if (offset > row)
			data[offset * W + row] = 0.0;
	}
	for (int row = 1; row < *height; ++row) {
		int H = row * W;
		if (threadIdx.x == 0)
			atomicAdd(lock, 1);
		if (offset > row) {
			for (int r = row - 1; r >= 0; --r)
				data[r * W + offset] -= data[H + offset] * data[r * W + row];
		}
		__syncthreads();
		if (threadIdx.x == 0) {
			atomicAdd(lock, -1);
			while (*lock != 0);
		}
		__syncthreads();
		if (offset < row)
			data[offset * W + row] = 0.0;
	}
}

// needed?
__global__ void rowReduce(float *data1, float *equs, float *identity, int *width, int *height, int *lock) {
	
}

__global__ void determinant(float *data, int *sign, float *joiner, int *size, int *lock) {
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ int W = *width; 
	float val = 0.0;
	int index = 0;
	if (offset == 0)
		*sign = 1;
	if (threadIdx.x == 0)
		atomicAdd(lock, 1);
	__syncthreads();
	if (threadIdx.x == 0) {
		atomicAdd(lock, -1);
		while (*lock != 0);
	}
	__syncthreads();
	for (int row = 0; row < *height; ++row) {
		int H = row * W;
		if (threadIdx.x == 0)
			atomicAdd(lock, 1);
		__syncthreads();
		if (threadIdx.x == 0) {
			if (offset == 0) {
				val = abs(data[r * W + row]);
				index = row;
				for (int r = row + 1; r < *height; ++r) {
					float abs = abs(data[r * W + row]);
					if (abs >= val) {
						index = r;
						val = abs;
					}
				}
				if (index != row) {
					val = data[H + row];
					data[H + row] = data[index * W + row];
					data[index * W + row] = val;
				}
			}
			atomicAdd(lock, -1);
			while (*lock != 0);
		}
		__syncthreads();
		if (data[H + row] == 0.0) {
			if (offset == 0)
				printf("determinant is zero, and no inverse exists", row, row);
			break;
		}
		if (index != row && offset > row + 1 && offset < W) {
			val = data1[H + offset];
			data1[H + offset] = data1[index * W + offset];
			data1[index * W + offset] = val;
		}
		if (threadIdx.x == 0)
			atomicAdd(lock, 1);
		if (offset > row)
			for (int r = row + 1; r < *height; ++r)
				data1[r * W + offset] -= data1[H + offset] * data1[r * W + row] / data1[H + row];
		__syncthreads();
		if (threadIdx.x == 0) {
			atomicAdd(lock, -1);
			while (*lock != 0);
		}
		__syncthreads();
	}
	__shared__ float c[threadsPerBlock];
	c[threadIdx.x] = data[offset];
	__syncthreads();
	for (int flag = 1; flag < blockDim.x; flag *= 2) {
		if (offset + flag < *len && threadIdx.x % (flag * 2) == 0)
			c[threadIdx.x] *= c[threadIdx.x + flag];
		__syncthreads();
	}
	joiner[blockIdx.x] = c[0];
	if (threadIdx.x == 0)
		atomicAdd(lock, 1);
	__syncthreads();
	if (threadIdx.x == 0) {
		atomicAdd(lock, -1);
		while (*lock != 0);
	}
	__syncthreads();
	if (offset == 0)
		for (int i = 0; i < (*size / threadsPerBlock) + 1; ++i)
			*sign *= joiner[i];
}

__global__ void adjunct(float *data, int *sign, float *joiner, float *identity, int *size, int *lock) {
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ int W = *width; 
	float val = 0.0;
	int index = 0;
	if (offset == 0)
		*sign = 1;
	if (threadIdx.x == 0)
		atomicAdd(lock, 1);
	__syncthreads();
	if (threadIdx.x == 0) {
		atomicAdd(lock, -1);
		while (*lock != 0);
	}
	__syncthreads();
	for (int row = 0; row < *height; ++row) {
		int H = row * W;
		if (threadIdx.x == 0)
			atomicAdd(lock, 1);
		__syncthreads();
		if (threadIdx.x == 0) {
			if (offset == 0) {
				val = abs(data[r * W + row]);
				index = row;
				for (int r = row + 1; r < *height; ++r) {
					float abs = abs(data[r * W + row]);
					if (abs >= val) {
						index = r;
						val = abs;
					}
				}
				if (index != row) {
					val = data[H + row];
					data[H + row] = data[index * W + row];
					data[index * W + row] = val;
					*sign = -*sign;
				}
			}
			atomicAdd(lock, -1);
			while (*lock != 0);
		}
		__syncthreads();
		if (data[H + row] == 0.0) {
			if (offset == 0)
				printf("determinant is zero, and no inverse exists", row, row);
			break;
		}
		if (index != row) {
			val = identity[H + offset];
			identity[H + offset] = identity[index * W + offset];
			identity[index * W + offset] = val;
			if (offset > row + 1 && offset < W) {
				val = data[H + offset];
				data[H + offset] = data[index * W + offset];
				data[index * W + offset] = val;
			}
		}
		if (threadIdx.x == 0)
			atomicAdd(lock, 1);
		if (offset > row) {
			for (int r = row + 1; r < *height; ++r)
				data[r * W + offset] -= data[H + offset] * data[r * W + row] / data[H + row];
		}
		for (int r = row + 1; r < *height; ++r)
			identity[r * W + offset] -= identity[H + offset] * data[r * W + row] / data[H + row];
		__syncthreads();
		if (threadIdx.x == 0) {
			atomicAdd(lock, -1);
			while (*lock != 0);
		}
		__syncthreads();
	}
	__shared__ float c[threadsPerBlock];
	c[threadIdx.x] = data[offset];
	__syncthreads();
	for (int flag = 1; flag < blockDim.x; flag *= 2) {
		if (offset + flag < *len && threadIdx.x % (flag * 2) == 0)
			c[threadIdx.x] *= c[threadIdx.x + flag];
		__syncthreads();
	}
	joiner[blockIdx.x] = c[0];
	if (threadIdx.x == 0)
		atomicAdd(lock, 1);
	__syncthreads();
	if (threadIdx.x == 0) {
		atomicAdd(lock, -1);
		while (*lock != 0);
	}
	__syncthreads();
	if (offset == 0) {
		for (int i = 0; i < (*size / threadsPerBlock) + 1; ++i)
			*sign *= joiner[i];
	}
	for (int col = row + 1; col < W; ++col)
		data[offset * W + col] /= data[offset * W + row];
	for (int col = 0; col < W; ++col)
		identity[offset * W + col] /= data[offset * W + row];
	data[offset * W + row] = 1.0;
	for (int row = 1; row < *height; ++row) {
		int H = row * W;
		if (threadIdx.x == 0)
			atomicAdd(lock, 1);
		if (offset > row) {
			for (int r = row - 1; r >= 0; --r)
				data[r * W + offset] -= data[H + offset] * data[r * W + row];
		}
		for (int r = row - 1; r >= 0; --r)
			identity[r * W + offset] -= identity[H + offset] * data[r * W + row];
		__syncthreads();
		if (threadIdx.x == 0) {
			atomicAdd(lock, -1);
			while (*lock != 0);
		}
		__syncthreads();
		if (offset < row)
			data[offset * W + row] = 0.0;
	}
	for (int r = 0; r < W; ++r)
		identity[r * W + offset] /= *sign;
}

__global__ void cofactor(float *data, int *sign, float *joiner, float *identity, int *size, int *lock) {
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ int W = *width; 
	float val = 0.0;
	int index = 0;
	if (offset == 0)
		*sign = 1;
	if (threadIdx.x == 0)
		atomicAdd(lock, 1);
	__syncthreads();
	if (threadIdx.x == 0) {
		atomicAdd(lock, -1);
		while (*lock != 0);
	}
	__syncthreads();
	for (int row = 0; row < *height; ++row) {
		int H = row * W;
		if (threadIdx.x == 0)
			atomicAdd(lock, 1);
		__syncthreads();
		if (threadIdx.x == 0) {
			if (offset == 0) {
				val = abs(data[r * W + row]);
				index = row;
				for (int r = row + 1; r < *height; ++r) {
					float abs = abs(data[r * W + row]);
					if (abs >= val) {
						index = r;
						val = abs;
					}
				}
				if (index != row) {
					val = data[H + row];
					data[H + row] = data[index * W + row];
					data[index * W + row] = val;
					*sign = -*sign;
				}
			}
			atomicAdd(lock, -1);
			while (*lock != 0);
		}
		__syncthreads();
		if (data[H + row] == 0.0) {
			if (offset == 0)
				printf("determinant is zero, and no inverse exists", row, row);
			break;
		}
		if (index != row) {
			val = identity[H + offset];
			identity[H + offset] = identity[index * W + offset];
			identity[index * W + offset] = val;
			if (offset > row + 1 && offset < W) {
				val = data[H + offset];
				data[H + offset] = data[index * W + offset];
				data[index * W + offset] = val;
			}
		}
		if (threadIdx.x == 0)
			atomicAdd(lock, 1);
		if (offset > row) {
			for (int r = row + 1; r < *height; ++r)
				data[r * W + offset] -= data[H + offset] * data[r * W + row] / data[H + row];
		}
		for (int r = row + 1; r < *height; ++r)
			identity[r * W + offset] -= identity[H + offset] * data[r * W + row] / data[H + row];
		__syncthreads();
		if (threadIdx.x == 0) {
			atomicAdd(lock, -1);
			while (*lock != 0);
		}
		__syncthreads();
	}
	__shared__ float c[threadsPerBlock];
	c[threadIdx.x] = data[offset];
	__syncthreads();
	for (int flag = 1; flag < blockDim.x; flag *= 2) {
		if (offset + flag < *len && threadIdx.x % (flag * 2) == 0)
			c[threadIdx.x] *= c[threadIdx.x + flag];
		__syncthreads();
	}
	joiner[blockIdx.x] = c[0];
	if (threadIdx.x == 0)
		atomicAdd(lock, 1);
	__syncthreads();
	if (threadIdx.x == 0) {
		atomicAdd(lock, -1);
		while (*lock != 0);
	}
	__syncthreads();
	if (offset == 0) {
		for (int i = 0; i < (*size / threadsPerBlock) + 1; ++i)
			*sign *= joiner[i];
	}
	for (int col = row + 1; col < W; ++col)
		data[offset * W + col] /= data[offset * W + row];
	for (int col = 0; col < W; ++col)
		identity[offset * W + col] /= data[offset * W + row];
	data[offset * W + row] = 1.0;
	for (int row = 1; row < *height; ++row) {
		int H = row * W;
		if (threadIdx.x == 0)
			atomicAdd(lock, 1);
		if (offset > row) {
			for (int r = row - 1; r >= 0; --r)
				data[r * W + offset] -= data[H + offset] * data[r * W + row];
		}
		for (int r = row - 1; r >= 0; --r)
			identity[r * W + offset] -= identity[H + offset] * data[r * W + row];
		__syncthreads();
		if (threadIdx.x == 0) {
			atomicAdd(lock, -1);
			while (*lock != 0);
		}
		__syncthreads();
		if (offset < row)
			data[offset * W + row] = 0.0;
	}
	for (int r = 0; r < W; ++r)
		identity[r * W + offset] /= *sign;
	if (threadIdx.x == 0)
		atomicAdd(lock, 1);
	__syncthreads();
	if (threadIdx.x == 0) {
		atomicAdd(lock, -1);
		while (*lock != 0);
	}
	__syncthreads();
	for (int r = offset + 1; r < W; ++r) {
		float temp = identity[r * W + offset];
		identity[r * W + offset] = identity[offset * W + r];
		identity[offset * W + r] = temp;
	}
}

__global__ void normalize(float *data, float *len, float *multiple, float *lock) {
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
		atomicAdd(&multiple, c[0]);
	if (threadIdx.x == 0)
		atomicAdd(lock, 1);
	__syncthreads();
	if (threadIdx.x == 0) {
		atomicAdd(lock, -1);
		while (*lock != 0);
	}
	__syncthreads();
	if (offset == 0)
		*multiple = sqrt(*multiple);
	if (threadIdx.x == 0)
		atomicAdd(lock, 1);
	__syncthreads();
	if (threadIdx.x == 0) {
		atomicAdd(lock, -1);
		while (*lock != 0);
	}
	__syncthreads();
	data[offset] /= *multiple;
}

// sum row. just use a funct call to sum with a cuda memcpy of the selected range

// sum col. just use a funct call to sum with a extractCol of the selected range

struct matrix {
	int width = -1, height = -1;
	float **data = nullptr;
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
		long dataSize = width * height * sizeof(float);
		float *ret, *input;
		int *len;
		float retVal = 0.0;
		int size = width * height;
		cudaMalloc(&ret, sizeof(float));
		cudaMalloc(&input, dataSize);
		cudaMalloc(&len, sizeof(int));
		cudaMemcpy(ret, &ret, sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(input, &data[0], dataSize, cudaMemcpyHostToDevice);
		cudaMemcpy(len, &size, sizeof(int), cudaMemcpyHostToDevice);
		int blocks = (vectSize + threadsPerBlock - 1) / threadsPerBlock;
		sum<<<blocks, threadsPerBlock>>>(input, ret, len);
		cudaMemcpy(&retVal, ret, sizeof(float), cudaMemcpyDeviceToHost);
		cudaFree(ret);
		cudaFree(input);
		cudaFree(len);
		return retVal;
	}

	~matrix() {
		delete data;
	}
}

int main() {

 	return 0;
}

