
#include <iostream>

#define N 10

__global__
void add_kernel(int *a, int *b, int *c) {
    for (int i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
    }

}

int main(int argc, char *argv[]) {

    int a[N];
    int b[N];
    int c[N];

    for (int i = 0; i < N; i++) {
        a[i] = -1;
        b[i] = i * i;
    }

    add_kernel(a, b, c);

    for (int i = 0; i < N; i++) {
        std::cout << a[i] << " + " << b[i] << " = " << c[i] << "\n";
    }

    
    return 0;
}
