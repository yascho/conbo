from setuptools import setup
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='conbo',
    version='0.1.1',
    description='A package for bounding properties of random variables with high confidence.',
    long_description=long_description,
    long_description_content_type="text/markdown",
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
