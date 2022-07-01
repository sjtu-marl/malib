
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
./install.sh
```

## Environments

MALib integrates many popular reinforcement learning environments, we list some of them as follows.

- [Google Research Football](https://github.com/google-research/football): RL environment based on open-source game Gameplay Football.
- [SMAC](https://github.com/oxwhirl/smac): An environment for research in the field of collaborative multi-agent reinforcement learning (MARL) based on Blizzard's StarCraft II RTS game.
- [Gym](https://github.com/openai/gym): An open source environment collections for developing and comparing reinforcement learning algorithms.
- [PettingZoo](https://github.com/Farama-Foundation/PettingZoo): Gym for multi-agent reinforcement learning.
- [OpenSpiel](https://github.com/deepmind/open_spiel): A framework for Reinforcement Learning in games, it provides plenty of environments for the research of game theory.

In addition, users can customize environments with MALib's environment interfaces. Please refer to our documentation.

## Algorithms

MALib integrates population-based reinforcement learning, classical multi-agent and single-agent reinforcement learning algorithms. See algorithms table [here](/algorithms.md).

## Quick Start

Before running examples, please ensure that you import python path as:

```bash
cd malib
export PYTHONPATH=./
```

- Training PSRO with running `python examples/run_psro.py`
- Training Gym example with running `python examples/run_gym.py`
- Training Google Research Football cases you can run `python examples/run_grfootball.py`. It runs single agent training by default, you can activate group training with `--use_group`.
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
