"""A setuptools based setup module for catstore.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""
USE_CYTHON = True

# Always prefer setuptools over distutils
from setuptools import setup, Extension, find_packages
# To use a consistent encoding
from codecs import open
import os
from os import path
import numpy

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

os.system("git describe && echo version = \\\"`git describe`\\\" > catstore/version.py")
from catstore import version


if USE_CYTHON:
    try:
        from Cython.Distutils import build_ext
    except ImportError:
        if USE_CYTHON=='auto':
            USE_CYTHON=False
        else:
            raise
            

cmdclass = {}
ext_modules = []

if USE_CYTHON:
    ext_modules += [
        Extension("catstore._querycat",
            [ "catstore/_querycat.pyx" ],
            include_dirs=[numpy.get_include()],
            libraries=['m'],
            #define_macros=[('CYTHON_TRACE', '1')],
            #extra_compile_args = ["-ffast-math","-O3"]
            #extra_compile_args=['-fopenmp'],
            #extra_link_args=['-fopenmp'],
            ),
    ]
    cmdclass.update({ 'build_ext': build_ext })
else:
    ext_modules += [
        Extension("catstore._querycat",
            [ "catstore/_querycat.c" ],
            include_dirs=[numpy.get_include()],
            libraries=['m']),
    ]

setup(
    name='catstore',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version=version.version,

    description='Interface full-sky astronomical catalogues',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/bengranett/catstore',

    # Author details
    author='Ben Granett, Dida Markovic',
    author_email='ben.granett@brera.inaf.it',

    # Choose your license
    license='MIT',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Scientists',
        'Topic :: Science :: Astronomy',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
    ],

    # What does your project relate to?
    keywords='astronomy',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),

    cmdclass = cmdclass,
    ext_modules=ext_modules,


    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    #   py_modules=["my_module"],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=['h5py', 'h5py_cache', 'numpy'],

    # # List additional groups of dependencies here (e.g. development
    # # dependencies). You can install these using the following syntax,
    # # for example:
    # # $ pip install -e .[dev,test]
    # extras_require={
    #     'dev': ['check-manifest'],
    #     'test': ['coverage'],
    # },

    # # If there are data files included in your packages that need to be
    # # installed, specify them here.  If using Python 2.6 or less, then these
    # # have to be included in MANIFEST.in as well.
    # package_data={
    #     'sample': ['package_data.dat'],
    # },

    # # Although 'package_data' is the preferred approach, in some case you may
    # # need to place data files outside of your packages. See:
    # # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files # noqa
    # # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    # data_files=[('my_data', ['data/data_file'])],

    # # To provide executable scripts, use entry points in preference to the
    # # "scripts" keyword. Entry points provide cross-platform support and allow
    # # pip to create the appropriate form of executable for the target platform.
    entry_points={
        'console_scripts': [
            'catstore=catstore.main:main',
        ],
    },
)
