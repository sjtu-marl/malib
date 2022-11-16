
<div align=center><img src="docs/imgs/logo.svg" width="35%"></div>


# MALib: A parallel framework for population-based reinforcement learning

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/sjtu-marl/malib/blob/main/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/malib/badge/?version=latest)](https://malib.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://app.travis-ci.com/sjtu-marl/malib.svg?branch=test-cases)](https://app.travis-ci.com/sjtu-marl/malib.svg?branch=test-cases)
[![codecov](https://codecov.io/gh/sjtu-marl/malib/branch/test-cases/graph/badge.svg?token=CJX14B2AJG)](https://codecov.io/gh/sjtu-marl/malib)

MALib is a parallel framework of population-based learning nested with reinforcement learning methods, such as Policy Space Response Oracle, Self-Play, and Neural Fictitious Self-Play. MALib provides higher-level abstractions of MARL training paradigms, which enables efficient code reuse and flexible deployments on different distributed computing paradigms.

![architecture](docs/imgs/Architecture.svg)

## Installation

The installation of MALib is very easy. We've tested MALib on Python 3.6 and 3.7. This guide is based on ubuntu 18.04 and above. We strongly recommend using [conda](https://docs.conda.io/en/latest/miniconda.html) to manage your dependencies, and avoid version conflicts. Here we show the example of building python 3.7 based conda environment.


```bash
conda create -n malib python==3.7 -y
conda activate malib

# install dependencies
./install.sh
```

## Environments

MALib integrates many popular reinforcement learning environments, we list some of them as follows.

- [OpenSpiel](https://github.com/deepmind/open_spiel): A framework for Reinforcement Learning in games, it provides plenty of environments for the research of game theory.
- [Gym](https://github.com/openai/gym): An open source environment collections for developing and comparing reinforcement learning algorithms.
- [Google Research Football](https://github.com/google-research/football): RL environment based on open-source game Gameplay Football.
- [SMAC](https://github.com/oxwhirl/smac): An environment for research in the field of collaborative multi-agent reinforcement learning (MARL) based on Blizzard's StarCraft II RTS game.

In addition, users can customize environments with MALib's environment interfaces. Please refer to our documentation.

## Algorithms

MALib integrates population-based reinforcement learning, classical multi-agent and single-agent reinforcement learning algorithms. See algorithms table [here](/algorithms.md).

## Quick Start

Before running examples, please ensure that you import python path as:

```bash
cd malib
export PYTHONPATH=./
```

- Running PSRO example: `python examples/run_psro.py`
- Running RL example: `python examples/run_gym.py`

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
