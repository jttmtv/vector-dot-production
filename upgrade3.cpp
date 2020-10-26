#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <book.h>
#define min(a,b) (a<b?a:b)

const int N = 200000000;
const int threadsPerBlock = 256;
const int blocksPerGrid = min(32, (N + threadsPerBlock - 1) / threadsPerBlock);


__global__ void dot(float *a, float *b, float *c) {
    __shared__ float cache[threadsPerBlock];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;
    /*Calculate a global offset tid in the input array through the thread block index and the thread index. The offset cacheIndex in the shared memory cache is equal to the thread index. 
    The thread block index has nothing to do with this offset, because each thread block has a private copy of the shared memory*/

    float temp = 0;
    while (tid < N) {/*Prevent indexes from crossing array boundaries*/
        temp += a[tid] * b[tid];
        /*After each thread calculates the task on the current index, it then needs to increment the index, where the increment step is the thread running in the grid
         The number of threads. This value is equal to the number of threads in each thread block multiplied by the number of thread blocks in the thread.*/
        tid += blockDim.x * gridDim.x;
    }

    /*Set the value at the corresponding position in the cache.*/
    cache[cacheIndex] = temp;

    /*Synchronize the threads in the thread block. Ensure that all write operations to the shared array cache[] are completed before reading the cache.*/
    __syncthreads();

    //For the reduction operation, the following code requires threadPerBlock to be an exponent of 2
    int i = blockDim.x / 2;
    while (i != 0){
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        /*在读取cache[]中的值之前，首先需要确保每个写入cache[]的线程都已经执行完毕。*/
        __syncthreads();
        i /= 2;
    }
    if (cacheIndex == 0)
        c[blockIdx.x] = cache[0];
    /*After the while loop is finished, each thread block gets a value, which is located in the first element of cache[]. Because there is only one value
     Write to global memory, so only one thread is needed to perform this operation. Of course, each thread can perform this write operation, but doing so will
     Makes unnecessary memory traffic when writing a single value. For simplicity, the thread with index 0 is selected. Finally, since each thread block is
     Only write a value to the global data c[], so this value can be indexed by blockIdx.*/
}

int main(){
    float *a, *b, c, *partial_c;
    float *dev_a, *dev_b, *dev_partial_c;

    //Initialize random number seed
    srand((unsigned)time(NULL));

    //Allocate memory on the CPU
    a = (float *)malloc(N*sizeof(float));
    b = (float *)malloc(N*sizeof(float));
    partial_c = (float *)malloc(blocksPerGrid*sizeof(float));

    //Allocate memory on the GPU
    HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_partial_c, blocksPerGrid * sizeof(float)));

    //Fill host memory
    for (int i = 0; i < N; i++){
		vec1[i] = rand() / float(RAND_MAX);
    vec2[i] = rand() / float(RAND_MAX);
	}

    //Copy the arrays'a' and'b' to the GPU
    HANDLE_ERROR(cudaMemcpy(dev_a, a, N*sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, N*sizeof(float), cudaMemcpyHostToDevice));
    dot << <blocksPerGrid, threadsPerBlock >> >(dev_a, dev_b, dev_partial_c);

    //Copy array'c' from GPU to CPU
    HANDLE_ERROR(cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid * sizeof(float),
        cudaMemcpyDeviceToHost));

    //Complete the final summation operation on the CPU
    c = 0.0f;
    for (int i = 0; i < blocksPerGrid; i++){
        c += partial_c[i];
    }

#define sum_squares(x) (x*(x+1)*(2*x+1)/6)
    printf("Does GPU value %.6g = %.6g?\n", c, 2 * sum_squares((float)(N - 1)));

    //Free up memory on the GPU
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_partial_c);

    //Free up memory on the CPU
    free(a);
    free(b);
    free(partial_c);
}
