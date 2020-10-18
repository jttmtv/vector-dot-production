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
    /*通过线程块索引和线程索引计算出输入数组中的一个全局偏移tid。共享内存缓存中的偏移cacheIndex
    就等于线程索引.线程块索引与这个偏移无关，因为每个线程块都拥有该共享内存的私有副本*/

    float temp = 0;
    while (tid < N) {/*防止索引越过数组边界*/
        temp += a[tid] * b[tid];
        /*在每个线程计算完当前索引上的任务后，接着就需要对索引进行递增，其中递增的步长为线程格中正在运行
        的线程数量。这个数值等于每个线程块中的线程数量乘以线程中线程块的数量。*/
        tid += blockDim.x * gridDim.x;
    }

    /*设置cache中相应位置上的值。*/
    cache[cacheIndex] = temp;

    /*对线程块中的线程进行同步。确保所有对共享数组cache[]的写入操作在读取cache之前完成。*/
    __syncthreads();

    //对于归约运算来说，以下代码要求threadPerBlock必须是2的指数
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
    /*在结束了while循环后，每个线程块都得到了一个值，这个值位于cache[]的第一个元素中。因为只有一个值
    写入到全局内存，因此只需要一个线程来执行这个操作。当然，每个线程都可以执行这个写入操作，但这么做将
    使得在写入单个值时带来不必要的内存通信量。为了简单，选择了索引为0的线程。最后，由于每个线程块都
    只写入一个值到全局数据c[]中，因此可以通过blockIdx来索引这个值。*/
}

int main(void){
    float *a, *b, c, *partial_c;
    float *dev_a, *dev_b, *dev_partial_c;

    //初始化随机数种子
    srand((unsigned)time(NULL));

    //在CPU上分配内存
    a = (float *)malloc(N*sizeof(float));
    b = (float *)malloc(N*sizeof(float));
    partial_c = (float *)malloc(blocksPerGrid*sizeof(float));

    //在GPU上分配内存
    HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_partial_c, blocksPerGrid * sizeof(float)));

    //填充主机内存
    for (int i = 0; i < N; i++){
		vec1[i] = rand() / float(RAND_MAX);
    vec2[i] = rand() / float(RAND_MAX);
	}

    //将数组'a'和'b'复制到GPU
    HANDLE_ERROR(cudaMemcpy(dev_a, a, N*sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, N*sizeof(float), cudaMemcpyHostToDevice));
    dot << <blocksPerGrid, threadsPerBlock >> >(dev_a, dev_b, dev_partial_c);

    //将数组'c'从GPU复制到CPU
    HANDLE_ERROR(cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid * sizeof(float),
        cudaMemcpyDeviceToHost));

    //在CPU上完成最终的求和运算
    c = 0;
    for (int i = 0; i < blocksPerGrid; i++){
        c += partial_c[i];
    }

#define sum_squares(x) (x*(x+1)*(2*x+1)/6)
    printf("Does GPU value %.6g = %.6g?\n", c, 2 * sum_squares((float)(N - 1)));

    //释放GPU上的内存
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_partial_c);

    //释放CPU上的内存
    free(a);
    free(b);
    free(partial_c);
}