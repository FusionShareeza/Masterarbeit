from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

# setup(
#     ext_modules=[
#         Extension("fun", ["fun.c"],
#                   include_dirs=[numpy.get_include()]),
#     ],
# )

setup(
    ext_modules=cythonize(
        ['funktion4.pyx', ],  # Python code file with primes() function
        annotate=True),                 # enables generation of the html annotation file
)