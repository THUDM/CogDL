from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name='dgNN',
    version='0.0.1',
    ext_modules=[
        CUDAExtension('fused_gat', ['fused_gat/fused_gat.cpp', 'fused_gat/fused_gat_kernel.cu']),
        CUDAExtension('fused_edgeconv',['fused_edgeconv/fused_edgeconv.cpp','fused_edgeconv/fused_edgeconv_kernel.cu']),
        CUDAExtension('fused_gmm',['fused_gmm/fused_gmm.cpp','fused_gmm/fused_gmm_kernel.cu']),
        CUDAExtension('mhsddmm',['sddmm/mhsddmm.cc','sddmm/mhsddmm_kernel.cu']),
        CUDAExtension('mhtranspose',['csr2csc/mhtranspose.cc','csr2csc/mhtranspose_kernel.cu'])
        ],
        
    cmdclass={'build_ext': BuildExtension},
    install_requires=['torch']
)