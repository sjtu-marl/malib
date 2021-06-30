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
        "ray==1.3.0",
        "pettingzoo",
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
    ],
    extras_require={
        "dev": [
            "pip==21.0.1",
            "black==20.8b1",
            "pytest",
            "sphinx",
            "sphinx-rtd-theme",
            "sphinxcontrib-apidoc",
        ],
        "envs": [
            # "vizdoom==1.1.8",
            # "git+https://github.com/oxwhirl/smac.git",
            "supersuit",
            "multi-agent-ale-py",
            "autorom",
        ],
    },
)
