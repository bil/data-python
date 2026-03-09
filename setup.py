"""
Setup script.
"""

from setuptools import setup, find_packages

setup(
    name="bil",
    version="0.0.1",
    packages=find_packages(include=["bil", "bil.*"]),
    install_requires=[
        "numpy",
        "pandas",
        "h5py",
        "requests",
        "PyYAML",
        "tqdm",
        "cloudpickle",
    ],
)
