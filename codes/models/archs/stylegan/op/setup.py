#!/usr/bin/env python3
import os
import torch

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cxx_args = ['-std=c++11']

nvcc_args = [
    '-gencode', 'arch=compute_50,code=sm_50',
    '-gencode', 'arch=compute_52,code=sm_52',
    '-gencode', 'arch=compute_60,code=sm_60',
    '-gencode', 'arch=compute_61,code=sm_61',
    '-gencode', 'arch=compute_70,code=sm_70',
]

setup(
    name='stylegan2_ops_cuda',
    ext_modules=[
        CUDAExtension('fused_bias_act_cuda', [
            'fused_bias_act.cpp',
            'fused_bias_act_kernel.cu'
        ], extra_compile_args={'cxx': cxx_args, 'nvcc': nvcc_args}),
        CUDAExtension('upfirdn2d_cuda', [
            'upfirdn2d.cpp',
            'upfirdn2d_kernel.cu'
        ], extra_compile_args={'cxx': cxx_args, 'nvcc': nvcc_args})
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
