import pybind11
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

ext_modules = [
    Extension(
        'src.gamma_func_cpp.gamma_incomp',
        sources=['src/gamma_func_cpp/gamma_incomp.cpp'],
        include_dirs=[pybind11.get_include()],
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
)
