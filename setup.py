"""
Setup script.
"""

from setuptools import setup, find_packages

setup(
    name="bil",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "numpy < 2.0.0",
        "pandas",
        "h5py",
        "requests",
        "urllib3",
        "PyYAML",
        "tqdm",
        "cloudpickle",
    ],
    extras_require={
        "test": ["pytest", "pytest-mock", "matplotlib"],
    },
    python_requires=">=3.9",
)
