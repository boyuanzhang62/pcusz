import numpy as np
import torch
import ctypes
from ctypes import *
from random import random

# create example tensors on GPU
input_tensor_gpu = torch.tensor([random() for i in range(1024 * 1024)], dtype=torch.float32).cuda()
compressed_tensor_gpu = torch.tensor([0 for i in range(1024 * 1024 * 4)], dtype=torch.uint8).cuda()
output_tensor_gpu = torch.tensor([0 for i in range(1024 * 1024)], dtype=torch.float32).cuda()

# compression, use as __compress = pycusz_compress()
# see run_pycusz as an example
def pycusz_compress():
    dll = ctypes.CDLL('./capi.so', mode=ctypes.RTLD_GLOBAL)
    func = dll.compress
    func.argtypes = [POINTER(c_float), POINTER(c_ubyte), c_int, c_float, POINTER(c_size_t)]
    func.restype = POINTER(c_void_p)
    return func

# decompression, use as __decompress = pycusz_decompress()
# see run_pycusz as an example
def pycusz_decompress():
    dll = ctypes.CDLL('./capi.so', mode=ctypes.RTLD_GLOBAL)
    func = dll.decompress
    func.argtypes = [POINTER(c_ubyte), POINTER(c_float), c_int, c_float, POINTER(c_size_t), POINTER(c_void_p)]
    return func

def run_pycusz(input, compressed, output, file_size, error_bound):
    # get input GPU pointer
    input_gpu_ptr = input.data_ptr()
    input_gpu_ptr = cast(input_gpu_ptr, ctypes.POINTER(c_float))

    # get output GPU pointer
    output_gpu_ptr = output.data_ptr()
    output_gpu_ptr = cast(output_gpu_ptr, ctypes.POINTER(c_float))

    # get compressed GPU pointer, the size of compressed tensor is hard to decide in advance
    # to use it safely, allocate a tensor the same size as the input, just like this example
    # Jiannan advises to use half of the input size
    compressed_gpu_ptr = compressed.data_ptr()
    compressed_gpu_ptr = cast(compressed_gpu_ptr, ctypes.POINTER(c_ubyte))

    __size_pointer = POINTER(c_size_t)
    # get the pointer points to compressed size 
    compressed_size_ptr = __size_pointer(c_size_t(0))

    __compress = pycusz_compress()
    # launch compress, return a necessary ptrs points to intermidiate configuration variables
    ptrs = __compress(input_gpu_ptr, compressed_gpu_ptr, c_int(file_size), c_float(error_bound), compressed_size_ptr)

    # compressed size (compressed_size_ptr.contents) is in the unit of bytes
    # compression ratio = file_size * 4 / compressed_size_ptr.contents
    # print(compressed_size_ptr.contents)

    # launch decomporess
    __decompress = pycusz_decompress()
    __decompress(compressed_gpu_ptr, output_gpu_ptr, c_int(file_size), c_float(error_bound), compressed_size_ptr, ptrs)

if __name__ == '__main__':
    run_pycusz(input_tensor_gpu, compressed_tensor_gpu, output_tensor_gpu, 1024 * 1024, 1e-4)
