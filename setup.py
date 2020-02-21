from setuptools import find_packages
from setuptools import setup
from os import path as os_path

this_directory = os_path.abspath(os_path.dirname(__file__))

def read_file(filename):
    with open(os_path.join(this_directory, filename), encoding='utf-8') as f:
        long_description = f.read()
    return long_description

def read_requirement(filename):
    return [line.strip() for line in read_file(filename).splitlines() if not line.startswith('#')]

setup(
    name="handyML",
    version="0.0.1b1",
    author="ITryagain",
    author_email="long452a@163.com",
    python_requires='>=3.6.0',
    description="A library for data science",
    long_description=read_file('README.md'),
    long_description_content_type="text/markdown",
    url="https://github.com/leo6033/handyML",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    install_requires=read_requirement('requirements.txt'),
)