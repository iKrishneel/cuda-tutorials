
#include <iostream>

// inform that the function should run on device instead of the host
__global__
void add_kernel(int a, int b, int *c) {
    *c = a + b;
}

int main(int argc, char *argv[]) {

    // get device info
    int count;
    cudaGetDeviceCount(&count);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    std::cout << "Device Count: " << count << "\n";
    std::cout << "Name: " << prop.name  << "\n";
    std::cout << "\t mem: " << prop.maxThreadsPerBlock  << "\n";
    
    
    int c;
    int *dev_c;
    
    // allocate memory on the device
    // returned pointer should not be dereferenced
    cudaMalloc((void**)&dev_c, sizeof(int));
    
    // used to send device code to device compiler
    // angle brackets denote arguments we plan to pass for the device
    add_kernel<<< 1, 1 >>> (2, 7, dev_c);
    cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "SUM: " << c  << "\n";

    cudaFree(dev_c);
    
    std::cout << "HELLO WORLD!"  << "\n";
    return 0;
}
