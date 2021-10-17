import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="cen",
    version="0.0.1.dev",
    packages=find_packages(),
    description="Contextual Explanation Networks.",
    long_description=read("README.md"),
    author="Maruan Al-Shedivat",
    author_email="maruan@alshedivat.com",
    url="https://github.com/alshedivat/cen",
    license="MIT",
    install_requires=[
        "hydra-core",
        "numpy>=1.15.0",
        "scipy>=1.0.0",
        "scikit-learn",
    ],
    extras_require={
        "tf": ["tensorflow==2.0.0"],
        "tf_gpu": ["tensorflow-gpu==2.0.0"],
        "transformers": ["transformers"],
    },
)
