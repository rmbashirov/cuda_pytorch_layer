from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ext_modules = [
    CUDAExtension(
        'cuda_pytorch_layer.cuda.mult', [
            'cuda_pytorch_layer/cuda/mult.cpp',
            'cuda_pytorch_layer/cuda/mult_kernel.cu',
        ])
]

setup(
    version='0.0',
    author='Renat Bashirov',
    author_email='rmbashirov@gmail.com',
    install_requires=["torch>=1.3"],
    packages=['cuda_pytorch_layer', 'cuda_pytorch_layer.cuda'],
    name='cuda_pytorch_layer',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension}
)



