Installation Guide
==================

Conda Environment
-----------------

We've tested MALib on Python 3.6 and 3.7

.. code-block:: shell

    conda create -n malib python==3.7 -y
    conda activate malib
    pip install -e .

    # for development
    pip install -e .[dev]


Docker
------


**Optional**: if you wanna use `alpha-rank <https://arxiv.org/abs/1903.01373>`_ to solve meta-game, install open-spiel with its `installation guides <https://github.com/deepmind/open_spiel>`_.
