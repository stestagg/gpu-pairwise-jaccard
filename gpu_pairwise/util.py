from numba import cuda

def check_cuda():
    if not cuda.is_available():
        raise ImportError('gpu_pairwise requires a CUDA compatible device, and correct CUDA installation')
