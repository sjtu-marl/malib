.. _installation:

Installation Guide
==================

The installation of MALib is very easy. We've tested MALib on Python 3.6 and 3.7. This guide is based on ubuntu 18.04 or above.


Conda Environment
-----------------

We strongly recommend using `conda <https://docs.conda.io/en/latest/miniconda.html>`_ to manage your dependencies, and avoid version conflicts. Here we show the example of building python 3.7 based conda environment.

.. code-block:: shell

    conda create -n malib python==3.7 -y
    conda activate malib

    # install dependencies
    ./install_deps.sh

    # install malib
    pip install -e .


External environments are integrated in MALib, such as StarCraftII and vizdoom, you can install them via ``pip install -e .[envs]``. For users who wanna contribute to our repository, run ``pip install -e .[dev]`` to complete the development dependencies, also refer the contributing guide.


Docker
------

We also provide a dockered environment to support cross-platform tasting, some docker files are provided in the `Docker Examples`_

**Optional**: if you wanna use `alpha-rank <https://arxiv.org/abs/1903.01373>`_ to solve meta-game, install open-spiel with its `installation guides <https://github.com/deepmind/open_spiel>`_.
