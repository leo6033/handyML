from setuptools import find_packages
from setuptools import setup

setuptools.setup(
    name="handyML ITryagain",
    version="0.0.1",
    author="ITryagain",
    author_email="long452a@163.com",
    description="A library for data science",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)