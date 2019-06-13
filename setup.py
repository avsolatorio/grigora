import os

from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "grigora.keras.preprocessing.cython_skipgrams",
        ["grigora/keras/preprocessing/cython_skipgrams.pyx"],
        include_dirs=[np.get_include()]
    ),
]

here = os.path.abspath(os.path.dirname(__file__))
README = open(os.path.join(here, "README.md")).read()

requires = [
    'numpy>=1.16.0'
]

setup(
    name="grigora",
    version="0.0.3",
    description="Optimised implementation of common deep learning preprocessing utilities.",
    long_description=README,
    author="Aivin V. Solatorio",
    author_email="avsolatorio@gmail.com",
    url="https://github.com/avsolatorio/grigora",
    license="MIT",
    keywords="NLP data mining modelling word2vec preprocessing tokenization skipgrams",
    packages=find_packages(),
    include_package_data=True,
    package_data={"grigora": ["standard_logging.ini"]},
    zip_safe=False,
    install_requires=requires,
    tests_require=['pytest', 'keras'],
    ext_modules = cythonize(extensions),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
)

# python setup.py sdist bdist_wheel
# twine upload dist/*tar.gz
