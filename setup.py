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
        "pettingzoo==1.19.0",
        "grpcio-tools",
        "protobuf3-to-dict",
        "pickle5",
        "torch",
        "tensorboardX",
        "tensorboard",
        "readerwriterlock",
        "nashpy==0.0.21",
        "pymongo",
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
    ],
    extras_require={
        "dev": [
            "black==20.8b1",
            "pytest",
            "sphinx",
            "sphinx-rtd-theme",
            "sphinxcontrib-apidoc",
            "pytest-profiling",
            "pytest-cov",
            "pytest-mock",
            "pytest-xdist",
        ],
    },
)
