#include <cuda_runtime.h>
#include <iostream>

int main(){
    int *ptr, N=20;
    int *b;
    int num = 20;
    ptr = &num;//points to a memory inside host(CPU)
    cudaMalloc(&ptr, N*sizeof(int)); //points to a memory inside device(GPU)
    
    cudaMemcpy(ptr, &num, sizeof(int), cudaMemcpyHostToDevice);
    //without the above line, while printing *addr, value comes as 0
    //This is because, the memory region ptr inside device memory was never initialized with the value 20
    //For this, we create a memory region num inside host memory, copy it to GPU memory region ptr

    cudaMallocManaged(&b, N*sizeof(int)); //unified memory

    int *addr;
    cudaMallocHost(&addr,N*sizeof(int)); //points to a memory inside host(CPU)
    cudaPointerAttributes attr;

    cudaPointerGetAttributes(&attr, ptr);
    std::cout<<"Address is at "<<attr.type<<std::endl; //returns 2 showing that the memory resides in GPU
    std::cout<<"Address of the variable is "<<&ptr<<std::endl;

    cudaPointerGetAttributes(&attr, &(*addr));
    std::cout<<"Address is at "<<attr.type<<std::endl; //returns 1 showing that the memory resides in host (CPU)
    std::cout<<"Address of the variable is "<<&addr<<std::endl;

    cudaPointerGetAttributes(&attr, &(*b));
    std::cout<<"Address is at "<<attr.type<<std::endl; //returns 3 showing that the memory resides in unified memory
    std::cout<<"Address of the variable is "<<&b<<std::endl;

    cudaMemcpy(addr,ptr,N*sizeof(int),cudaMemcpyDeviceToHost);
    cudaPointerGetAttributes(&attr, addr);
    std::cout<<"After copying mem from device to host"<<std::endl;
    std::cout<<"Address is at "<<attr.type<<std::endl; //returns 2 showing that the memory resides in GPU
    std::cout<<"Address of the variable is "<<&addr<<std::endl;

    //std::cout<<"Value is "<<*ptr<<std::endl; 
    //Throws segmentation fault since ptr now points to a memory region inside GPU address space, de-refercing it 
    //tells the CPU to de-reference a memory region outside CPU space which actually doesn't exist
    //De-referencing of memory regions inside GPU(device) cannot be done from CPU(host).
    //The memory regions inside GPU needs to be moved to CPU to be seen.


    std::cout<<"Value is "<<*addr<<std::endl; 


    cudaFree(ptr);
    cudaFree(b);
    cudaFree(addr);
    
    return 0;
}