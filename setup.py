#!/usr/bin/env python

"""The setup script."""
import os
import numpy as np

from setuptools import setup, find_packages, Extension

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['dill', 'numba', 'numpy', 'scipy']

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest>=3', ]


try:
    from Cython.Build import cythonize
except ImportError:
    cythonize = None


# https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#distributing-cython-modules
def no_cythonize(extensions, **_ignore):
    for extension in extensions:
        sources = []
        for sfile in extension.sources:
            path, ext = os.path.splitext(sfile)
            if ext in (".pyx", ".py"):
                if extension.language == "c++":
                    ext = ".cpp"
                else:
                    ext = ".c"
                sfile = path + ext
            sources.append(sfile)
        extension.sources[:] = sources
    return extensions


extensions = [
    Extension(name="bof.cython.count", sources=["bof/cython/count.pyx"], language="c++",
              include_dirs=[np.get_include(), "."]),
]


if cythonize is not None:
    compiler_directives = {"language_level": 3, "embedsignature": True}
    extensions = cythonize(extensions, compiler_directives=compiler_directives)
else:
    extensions = no_cythonize(extensions)



# cfunc = cythonize(Extension(name="bof.cython.count", sources=["bof/cython/count.pyx"],
#                             include_dirs=[np.get_include(), "."]),
#                             compiler_directives = {"language_level": 3, "embedsignature": True})

setup(
    author="Fabien Mathieu",
    author_email='loufab@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Bag of Factors allow you to analyze a corpus from its self_factors.",
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/x-rst',
    include_package_data=True,
    keywords='bof',
    name='bof',
    packages=find_packages(),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/balouf/bof',
    version='0.3.1',
    zip_safe=False,
    ext_modules=extensions
)
