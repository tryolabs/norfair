from setuptools import Extension

import Cython.Build
import numpy


extension = Extension(
    "optimized_iou",
    sources=["norfair/optimized_iou.pyx"],
    include_dirs=[numpy.get_include()],
)


def build(setup_kwargs):
    """
    Build the optimized IoU distance module implemented in Cython.

    Parameters
    ----------
    setup_kwargs : Any
        This parameter is required to be able to build with Poetry.
    """
    setup_kwargs.update(
        {
            # declare the extension so that setuptools will compile it
            "ext_modules": Cython.Build.cythonize(extension, language_level=3),
        }
    )
