# MIT License

# Copyright (c) 2021 MARL @ SJTU

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from os import path
from setuptools import setup, find_packages

DIR = path.abspath(path.dirname(__file__))
with open(path.join(DIR, "README.md"), encoding="utf-8") as f:
    long_desc = f.read()


setup(
    name="MALib",
    description="A Parallel Framework for Population-based MARL",
    long_description=long_desc,
    long_description_content="text/markdown",
    version="0.1.0",
    packages=find_packages(exclude="tests"),
    include_packages_data=True,
    zip_safe=True,
    python_requires=">=3.7",
    install_requires=[
        "wrapt",
        "ray==1.13.0",
        "pickle5",
        "torch",
        "tensorboardX",
        "tensorboard",
        "readerwriterlock",
        "nashpy==0.0.21",
        "psutil",
        "pyecharts",
        "open_spiel>=1.0.2",
        "supersuit==3.3.1",
        "multi-agent-ale-py==0.1.11",
        "autorom==0.4.2",
        "colorlog==6.6.0",
        "mujoco_py",
        "hiredis==2.0.0",
        "frozendict==2.3.0",
        "numba>=0.56.0",
        "matplotlib>=3.5.3",
        "gym==0.23.0",
        "h5py==3.7.0",
        "pygame==2.1.0",
        "pettingzoo",
        "networkx",
    ],
    extras_require={
        "dev": [
            "black==22.3.0",
            "pytest",
            "sphinx",
            "sphinx-rtd-theme",
            "sphinxcontrib-apidoc",
            "sphinxcontrib-bibtex",
            "pytest-profiling",
            "pytest-cov",
            "pytest-mock",
            "pytest-xdist",
            "blackhc.mdp",
        ],
    },
)
