
<div align=center><img src="docs/imgs/logo.svg" width="35%"></div>


# MALib: A parallel framework for population-based multi-agent reinforcement learning

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/sjtu-marl/malib/blob/main/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/malib/badge/?version=latest)](https://malib.readthedocs.io/en/latest/?badge=latest)

MALib is a parallel framework of population-based learning nested with (multi-agent) reinforcement learning (RL) methods, such as Policy Space Response Oracle, Self-Play and Neural Fictitious Self-Play. MALib provides higher-level abstractions of MARL training paradigms, which enables efficient code reuse and flexible deployments on different distributed computing paradigms. The design of MALib also strives to promote the research of other multi-agent learning, including multi-agent imitation learning and model-based MARL.

![architecture](docs/imgs/Architecture.svg)

## Installation

The installation of MALib is very easy. We've tested MALib on Python 3.6 and 3.7. This guide is based on ubuntu 18.04 and above. We strongly recommend using [conda](https://docs.conda.io/en/latest/miniconda.html) to manage your dependencies, and avoid version conflicts. Here we show the example of building python 3.7 based conda environment.


```bash
conda create -n malib python==3.7 -y
conda activate malib

# install dependencies
./install_deps.sh

# install malib
pip install -e .
```

External environments are integrated in MALib, such as [StarCraftII](https://github.com/oxwhirl/smac) and [Mujoco](https://mujoco.org/). You can intall them by following the official guides on their project homepage.

## Quick Start

[TODO]

## Documentation

See [MALib Docs](https://malib.readthedocs.io/)

## Citing MALib


If you use MALib in your work, please cite the accompanying [paper](https://arxiv.org/abs/2106.07551).

```bibtex
@misc{zhou2021malib,
      title={MALib: A Parallel Framework for Population-based Multi-agent Reinforcement Learning}, 
      author={Ming Zhou and Ziyu Wan and Hanjing Wang and Muning Wen and Runzhe Wu and Ying Wen and Yaodong Yang and Weinan Zhang and Jun Wang},
      year={2021},
      eprint={2106.07551},
      archivePrefix={arXiv},
      primaryClass={cs.MA}
}
```
