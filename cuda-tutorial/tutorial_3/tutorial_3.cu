
#include <iostream>
#include <ctime>

#define N 50000


const int threads_per_block = 256;

__global__
void dot_gpu(float *a, float *b, float *c) {
    __shared__
        float cache[threads_per_block];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    float temp = 0;
    while (tid < N) {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }
    cache[cacheIndex] = temp;

}

int main(int argc, char *argv[]) {

    std::clock_t start_t;
    double duration;
    start_t = std::clock();
    
    int a[N];
    int b[N];
    int c[N];

    int *dev_a;
    int *dev_b;
    int *dev_c;

    // gpu timer
    cudaEvent_t start;
    cudaEventCreate(&start);
    cudaEventRecord(start, 0);
    
    // allocate memory on GPU
    cudaMalloc((void**)&dev_a, N * sizeof(int));
    cudaMalloc((void**)&dev_b, N * sizeof(int));
    cudaMalloc((void**)&dev_c, N * sizeof(int));
    
    for (int i = 0; i < N; i++) {
        a[i] = -1;
        b[i] = i * i;
    }

    // copy 2 arrays to device memory
    cudaMemcpy(dev_a, a, N * sizeof(N), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * sizeof(N), cudaMemcpyHostToDevice);


    // copy from device to host
    cudaMemcpy(c, dev_c, N * sizeof(N), cudaMemcpyDeviceToHost);

    // for (int i = 0; i < N; i++) {
    //     std::cout << a[i] << " + " << b[i] << " = " << c[i] << "\n";
    // }

    // add_cpu(a, b, c);

    cudaEvent_t stop;
    cudaEventCreate(&stop);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float dev_time;
    cudaEventElapsedTime(&dev_time, start, stop);
    std::cout << "Time: " << dev_time << "\n";
    
    std::cout << "DONE" << "\n";
    duration = ( std::clock() - start_t ) / (double) CLOCKS_PER_SEC;
    std::cout<<"printf: "<< duration <<'\n';
    
    
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    
    return 0;
}
