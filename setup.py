"""A setuptools based setup module.
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='dillinger',

    version='1.0.0.dev1',

    description='Bayesian optimization for iterated multi-armed bandit \
    experiments.',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/chipfranzen/dillinger',

    # Author details
    author='Charles Franzen',
    author_email='chip.franzen@gmail.com',

    license='MIT',

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6'
    ],

    # What does your project relate to?
    keywords='mutli-armed-bandits bayesian-optimization gaussian-processes',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=[]),

    install_requires=['numpy', 'scipy', 'matplotlib', 'seaborn', 'pandas'],
    python_requires='>=3',

)
