
#include <iostream>
#include <ctime>
#include <stdio.h>

#define N 500000

__global__
void add_kernel(int *a, int *b, int *c) {
    // blockIdx contains the value of the block index of the block
    // running
    // blockIdx can be defined in 2 dim
    int i = blockIdx.x;  // built in variables defined by cuda

    printf("Index: %d\n", i);
    
    if (i < N) {
        c[i] = a[i] + b[i];
    }
}

__host__
void add_cpu(int *a, int *b, int *c) {
    for (int i = 0; i < N; i++) {
        c[i] = a[i] + b[i];

        printf("%d\n", c[i]);
    }

}

__host__
void deviceBasics() {
    int d_count;
    cudaGetDeviceCount(&d_count);
    std::cout << "Device #" << d_count  << "\n";
    
    cudaDeviceProp d_prop;
    cudaGetDeviceProperties(&d_prop, 0);
    std::cout << d_prop.maxTexture3D[2] * 3 << "\n";
    
}

int main(int argc, char *argv[]) {

    deviceBasics();
    
    std::clock_t start;
    double duration;
    start = std::clock();
    
    int a[N];
    int b[N];
    int c[N];

    int *dev_a;
    int *dev_b;
    int *dev_c;

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

    // <<< first element is the # of parallel blocks to launch
    // second >>> the # of threads per block
    add_kernel<<<N, 1>>>(dev_a, dev_b, dev_c);

    // copy from device to host
    cudaMemcpy(c, dev_c, N * sizeof(N), cudaMemcpyDeviceToHost);

    // for (int i = 0; i < N; i++) {
    //     std::cout << a[i] << " + " << b[i] << " = " << c[i] << "\n";
    // }

    // add_cpu(a, b, c);
    
    std::cout << "DONE" << "\n";
    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    std::cout<<"printf: "<< duration <<'\n';
    
    
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    
    return 0;
}
