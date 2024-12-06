from setuptools import setup

setup(
    name='conbo',
    version='0.1.0',
    description='A package for bounding properties of random variables with high confidence',
    url='https://github.com/yascho/conbo',
    author='Yan Scholten',
    author_email='yan.scholten@tum.de',
    license='MIT',
    packages=['conbo'],
    install_requires=[
        'statsmodels==0.14.4',
        'numpy>=1.24.4'
    ],
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.11',
    ],
)
